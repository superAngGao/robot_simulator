Initiative: q54-optical-executor-algorithm-path
Stage: review
Author: claude
Version: v1
Date: 2026-04-30
Status: accepted-with-next-step
Related Files: collab/q54-optical-executor-algorithm-path__research__codex__v1.md, sensing/optical.py, optics/execution.py
Owner Summary: Claude accepted the seven-level executor capability path. The next implementation step should be L1 image-shaped depth/range/segmentation camera semantics, with camera ray generation and projected depth postprocessing in `sensing/`, while keeping executors ray-batch based.

# Q54 Optical Executor Algorithm Path Review

## Conclusion

The research quality is high and the external references are accurate. The
Drake, MuJoCo, Habitat-Sim, and Open3D patterns were extracted at the right
level and do not overfit unrelated rendering features.

The seven-level capability split and recommended sequence are accepted:

```text
L0 first-hit range/material/instance
L1 image-shaped depth/range/segmentation
L2 CPU acceleration
L3 direct-light/simple RGB
L4 raster renderer-style camera backend
L5 GPU / Q52 device result path
L6 Mitsuba/offline/high-fidelity/volume
```

The next implementation step is:

```text
sensing.OpticalPinholeCameraSpec
camera ray builder
image-shaped schema tests
depth_m postprocessor
```

## Q1. Is L1 the right next step?

Yes.

L0 already has `range_m` and `instance_id`, but the result is a flat ray batch.
This is not very useful for RL, Rerun, or debugging. L1 converts the same
executor output into image-shaped observations. It is the lowest-cost,
highest-value upgrade.

Skipping L1 and going directly to L3 direct-light RGB would introduce three
orthogonal problems at once:

```text
lighting model
image shape
camera semantics
```

Those should be debugged separately.

## Q2. Where should camera ray generation live?

Camera ray generation should live in `sensing/`.

The executor should stay ray-batch based:

```text
sensing.OpticalPinholeCameraSpec
  -> sensing.build_camera_rays(spec)
  -> OpticalRaySensorSpec
  -> OpticalExecutor.execute(snapshot, ray_spec)
```

The executor does not need camera-specific tests. Camera semantics should be
tested in `sensing/`.

## Q3. Where should projected `depth_m` be produced?

Projected `depth_m` should be produced by a camera-side postprocessor, not by a
camera-specific executor.

The transform is pure geometry:

```text
depth_m = range_m * dot(ray_direction, optical_axis)
```

Putting this in the executor would force every backend to reimplement it.
Keeping it in `sensing/` makes it common to CPU reference, Embree, OptiX, and
future backends.

## Q4. Should Open3D be an L2 backend?

Skip Open3D as an implementation backend for now.

Open3D's `RaycastingScene` is a good reference, but the package is heavy if all
we need is raycasting. L2 should decide between:

```text
Embree Python binding
in-repo simple BVH
```

If an Embree binding can be introduced cleanly, prefer it. Otherwise implement a
simple in-repo AABB tree / BVH.

## Q5. Should direct-light RGB wait for CPU acceleration?

Yes.

Direct-light RGB needs shadow rays, which means at least a second intersection
query. Without acceleration, the executor remains `O(num_rays * num_triangles)`
and shadow rays make it practical only for toy scenes.

## Q6. Should raster backends enter `OpticalExecutor` now?

No. Keep raster backends in `rendering/` until Q54 result contracts need them.

Raster backends produce framebuffer textures and need a GPU-memory to
`OpticalComputeResult.channels` bridge. That bridge is unstable until the Q52
device result path is connected. L4 should not enter the `OpticalExecutor`
interface before L5 device result ownership is clearer.
