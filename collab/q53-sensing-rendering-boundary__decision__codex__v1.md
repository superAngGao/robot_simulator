Initiative: q53-sensing-rendering-boundary
Stage: decision
Author: codex
Version: v1
Date: 2026-04-27
Status: accepted-review-followup-applied
Related Files: OPEN_QUESTIONS.md#Q53, collab/sensing-phase1__decision__codex__v1.md, physics/publish.py, rendering/render_scene.py, rendering/scene_builder.py, sensing/state_sample.py
Owner Summary: Q53 decision for non-numeric sensor boundaries. `sensing/` remains the owner of sensor-facing specs/readings and state/query views, `rendering/` remains the owner of renderable scene/backend primitives, and future render-backed camera/imaging execution should live in an integration layer rather than creating a direct `sensing -> rendering` dependency.

---

## 1. Decision Summary

Q53 is accepted with this boundary:

1. `sensing/` owns sensor-facing APIs:
   - numeric/state readings
   - future surface-query specs/results
   - sensor selection, calibration, noise, and observation-facing builders

2. `rendering/` owns renderable scene and backend APIs:
   - `RenderScene`
   - render scene builders
   - visualization / render backends
   - future render-world or rasterization primitives

3. A future `sensor_rendering/` integration layer should own render-backed
   imaging execution:
   - camera / RGB / segmentation / depth-image sensors
   - mapping sensor specs to render backend calls
   - converting rendered images into sensing readings

4. Do not introduce a direct `sensing -> rendering` dependency for phase-2.

This keeps `RenderScene` from becoming the universal sensor world model and
keeps `sensing/` from becoming a rendering frontend.

---

## 2. Dependency Direction

Accepted dependency shape:

```text
physics
  ├── rendering
  ├── sensing
  └── sensor_rendering   (future integration package)
        ├── rendering
        └── sensing
```

Rules:

- `physics/` does not import `rendering/`, `sensing/`, or `sensor_rendering/`.
- `sensing/` may consume published physics frames and telemetry bridges.
- `rendering/` may consume published physics frames to build visual scenes.
- `sensing/` should not import `rendering/`.
- `rendering/` should not need to import `sensing/` for visualization.
- `sensor_rendering/` may depend on both because its job is integration.

Rationale:

- Camera-like sensors need rendering machinery, but numeric sensors do not.
- Forcing camera support into `sensing/` would make all sensor code inherit a
  render dependency.
- Forcing camera readings into `RenderScene` would turn a debug/inspection scene
  into a general sensor packet.

---

## 3. ImagingView Ownership

Decision:

- Do not put `ImagingView` directly in `sensing/`.
- Do not add camera/image sensor payloads directly to `RenderScene`.
- Introduce a future integration package, tentatively `sensor_rendering/`, when
  camera or render-backed depth sensors become implementation work.

Proposed split:

- `sensing/`
  - owns `CameraSpec`, `ImageReading`, timestamps, env/body attachment metadata,
    calibration/noise policy, and observation-facing image conventions.
- `rendering/`
  - owns renderable scene/backend primitives and image generation capabilities.
- `sensor_rendering/`
  - owns `CameraSpec + PublishedFrame/RenderWorld -> ImageReading`.

Deferred:

- exact camera reading schema
- GPU realtime renderer and camera pipeline surface-cache sharing

Review follow-up:

- Use top-level `sensor_rendering/` rather than `rendering/sensors/` when this
  integration layer becomes real code. It depends on both `sensing/` and
  `rendering/`, so it should not appear to be owned by either side.
- First depth-image implementation should start from surface query / ray-cast
  depth, not render-backed imaging. RGB/segmentation remain the first clear
  `sensor_rendering/` use cases.

---

## 4. SurfaceQueryView Boundary

Decision:

- `SurfaceQueryView` belongs under `sensing/`, but it should not execute queries
  inside the view builder.
- The builder should produce a backend-neutral query view/spec.
- CPU/GPU query execution belongs in an explicit query runtime/executor layer.
- Until an executor exists, prefer the name `SurfaceQuerySpec` over
  `SurfaceQueryView`.

Reasoning:

- LiDAR, range finders, proximity sensors, and ray/depth probes are sensor
  concepts, so their specs/results belong in `sensing/`.
- CPU and GPU execution differ materially:
  - CPU likely uses host geometry traversal / BVH / numpy helpers.
  - GPU likely uses Warp kernels and published device-side surface buffers.
- Hiding execution inside a builder would blur frame materialization, query
  launch, synchronization, and result ownership.

Proposed phase-2 shape:

```text
sensing/
  surface_query.py       # SurfaceQuerySpec/View/Result dataclasses
  surface_builders.py    # build query specs from published frames + sensor specs
  surface_executor.py    # interface/protocol for executing queries

future backend-specific implementations:
  sensing/backends/cpu_surface_query.py
  sensing/backends/gpu_surface_query.py
```

The exact filenames are not committed by this decision; the important part is
the separation between query description and query execution.

---

## 5. RenderScene Relationship

`RenderScene` remains a debug/inspection renderable snapshot.

Do not use it as:

- the canonical geometry query scene for LiDAR
- the canonical camera world model
- a container for every sensor output

It may still be useful for debug visualization of sensor results, for example:

- drawing rays
- drawing hit points
- overlaying camera frustums
- displaying sampled contact/force annotations

Those overlays should remain visualization outputs, not the authoritative
sensor execution contract.

---

## 6. Impact On Earlier Questions

Q50 Step 4:

- Numeric/state sensor data can now proceed through `sensing` readings without
  waiting for imaging/query decisions.
- `RenderScene.sensor_data` should stay narrow if implemented; it should not
  become the general camera/LiDAR packet.

Q51:

- The force/telemetry part is partially unblocked by sensing phase-1.
- Imaging/query surfaces remain out of Q51 scope.

Q52:

- Future GPU publish policies should keep separate knobs for render-facing,
  state-sensing, surface-query, and imaging materialization.
- `PublishPolicy.sensor_render` / `PublishPlan.do_sensor_render` were renamed to
  `render_backed_sensing` / `do_render_backed_sensing` so the control plane does
  not imply that every sensor consumes rendering.

---

## 7. Review Follow-Up Decisions

Claude review accepted these follow-ups:

1. Use top-level `sensor_rendering/`.
2. Prefer `SurfaceQuerySpec` until query execution exists.
3. Treat first depth-image work as surface query / ray-cast depth; reserve
   `sensor_rendering/` for render-backed RGB/segmentation-style imaging.
4. Rename publish control-plane fields from `sensor_render` to
   `render_backed_sensing`.
