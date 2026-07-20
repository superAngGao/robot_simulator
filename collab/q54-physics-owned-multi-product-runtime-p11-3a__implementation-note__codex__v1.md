# Q54 Physics-Owned Multi-Product Runtime P11.3a Implementation Note

Author: Codex
Date: 2026-07-20
Status: implemented locally, not pushed

## Summary

Implemented P11.3a: removed the P11 product resolver dependency on
`go2_backend.py`.

P11.3 originally resolved `"video"` by reaching into
`go2_backend._build_video_camera`, `go2_backend._pack_video_rgb8`, and
`go2_backend.wp`. That violated the Go2 backend exit amendment: Go2 should be a
scene/preset/example, not a hidden backend dependency for the P11 user-facing
workflow.

This slice extracts the reviewed video defaults into generic lab modules and
keeps `preset_products.py` pointed at those components.

## Code Changes

- Added `tools/optical_pipeline_lab/camera_presets.py`.
  - `build_model_bounds_camera(...)` builds a fixed camera from scene bounds.
  - `build_lab_video_camera(...)` provides the reviewed fixed/orbit video camera
    builder used by the physics body-triangle video product.
  - The module owns its camera math directly and does not import
    `examples/` or `go2_backend.py`.
- Added `tools/optical_pipeline_lab/video_products.py`.
  - `create_physics_body_triangle_video_product_spec()` creates the reviewed
    `VideoProductSpec` for `physics_body_triangle_video_smoke`.
  - `synchronize_ready_event(...)` lazily imports Warp only when synchronization
    is actually requested.
  - Uses generic `video_loop.pack_video_rgb8` directly.
- Updated `tools/optical_pipeline_lab/preset_products.py`.
  - Removed the lazy import of `go2_backend`.
  - Registered `"video"` through
    `create_physics_body_triangle_video_product_spec`.
- Updated `MANIFEST.md` with the new module entries.

## Tests

Updated focused coverage in `tests/unit/optics/test_optical_pipeline_lab.py`:

- `test_resolve_lab_product_specs_builds_reviewed_video_and_debug_specs`
  - verifies `"video"` resolves to `VideoProductSpec`;
  - verifies the video spec uses `camera_presets.build_lab_video_camera`;
  - verifies RGB packing comes from `video_loop.pack_video_rgb8`;
  - verifies synchronization comes from `video_products.synchronize_ready_event`.
- `test_resolve_lab_product_specs_does_not_import_go2_backend`
  - runs product resolution in a fresh subprocess;
  - asserts resolving `"video"` does not load
    `tools.optical_pipeline_lab.go2_backend`;
  - asserts the resolver also does not load
    `examples.mujoco_menagerie_robot_preview`.

Existing P11.3 tests still cover explicit product pass-through, `"observation"`
fail-fast, unknown product fail-fast, and unsupported video preset fail-fast.

## Verification

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/camera_presets.py \
  tools/optical_pipeline_lab/video_products.py \
  tools/optical_pipeline_lab/preset_products.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/camera_presets.py \
  tools/optical_pipeline_lab/video_products.py \
  tools/optical_pipeline_lab/preset_products.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
All checks passed
4 files already formatted
```

Focused optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "resolve_lab_product_specs or camera_presets or video_products"
```

Result:

```text
6 passed, 164 deselected
```

Full optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
170 passed
```

## Boundaries

- `go2_backend.py` was not deleted in this slice.
- Existing legacy Go2 backend tests and CLI behavior were not rewritten.
- `go2_backend._build_video_camera` was not removed yet; legacy Go2 paths still
  use it.
- `run_optical_lab_preset(...)` was not added; that remains P11.4.
- Static Go2 P11 workflow support was not added.

## Notes

This is the first concrete step toward the Go2 backend exit standard:

```text
P11 product resolver no longer imports or references go2_backend.py.
```

The remaining retirement work is broader and should happen through the later
P11/P12 slices described in the design amendment.
