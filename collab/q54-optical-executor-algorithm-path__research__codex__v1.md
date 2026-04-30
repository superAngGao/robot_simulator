Initiative: q54-optical-executor-algorithm-path
Stage: research
Author: codex
Version: v1
Date: 2026-04-30
Status: draft-for-review
Related Files: optics/execution.py, optics/scene.py, optics/registry.py, collab/q54-optical-computation-workflow__discussion__codex__v1.md
Owner Summary: This document surveys open-source / open-documentation simulator and rendering stacks to choose a staged Q54 optical executor path. The recommended path is to keep the current first-hit reference executor, next add image-shaped depth/segmentation camera semantics, then add CPU acceleration, then direct-light/RGB, then GPU/Q52, and finally offline/volume paths.

# Q54 Optical Executor Algorithm Path Research

## 1. Research Question

Now that Q54 has:

```text
OpticalWorldRegistry
OpticalFrameInputs
OpticalSceneCache / OpticalSceneSnapshot
OpticalExecutor
OpticalComputeResult
```

we need to decide the algorithm path for real optical computation.

The risk is accidentally growing the reference first-hit executor into a
non-replaceable renderer. The goal is to define capability levels and backend
paths so future work can be sequenced and reviewed.

## 2. External Patterns

### Drake

Drake separates scene graph / render engine / sensors. Its `RenderEngine`
documentation exposes color, depth, and label images and explicitly distinguishes
projected image depth from laser range. This supports the Q54 decision to keep
`depth_m` and `range_m` as distinct channels.

Useful pattern:

```text
SceneGraph roles/labels -> RenderEngine -> RGB/depth/label image outputs
```

Relevance to Q54:

- label/segmentation is a first-class output, not an afterthought;
- camera depth should be projected depth, not ray range;
- range-style sensors need separate semantics.

Reference:

- https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1render_1_1_RenderEngine.html

### MuJoCo

MuJoCo's visualization stack is explicitly two-stage:

```text
mjModel + mjData -> mjvScene
mjvScene -> mjr_render / OpenGL framebuffer
```

The docs state the abstract visualization stage populates `mjvScene`, and the
OpenGL rendering stage renders that scene. This mirrors our desired split:
scene/cache prepares executable scene data; executor/backend computes results.

Robosuite/MuJoCo usage exposes RGB, depth, and segmentation as ground-truth
vision modalities.

Useful pattern:

```text
abstract scene construction separate from render backend
offscreen framebuffer for image observations
RGB/depth/segmentation as practical robotics outputs
```

References:

- https://mujoco.readthedocs.io/en/2.3.7/programming/visualization.html
- https://docs.hello-robot.com/0.3/stretch-mujoco/third_party/robosuite/docs/modules/renderers/

### Gazebo / gz-rendering

Gazebo keeps rendering as a library with pluggable render-engine support and
sensor integrations, including camera, lidar/gpu-rays, thermal, and visibility
mask behavior. This reinforces that sensor-specific outputs should be routed
through explicit sensor/executor paths rather than one generic renderer function.

Useful pattern:

```text
rendering engine plugins
sensor-specific passes
visibility masks for lidar/sensor outputs
```

Reference:

- https://gazebosim.org/libs/rendering/

### Habitat-Sim

Habitat-Sim exposes visual sensor specs with RGB/depth/semantic-like sensor
types and semantic target options such as semantic id vs object id. This is
close to our `roles`, `instance_id`, `semantic_id`, and image-shaped output
needs.

Useful pattern:

```text
sensor spec selects visual modality
semantic output can target semantic id or object id
scene graph remains separate from sensor observation
```

References:

- https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.CameraSensorSpec.html
- https://aihabitat.org/docs/habitat-sim/classesp_1_1gfx_1_1Renderer.html

### Open3D RaycastingScene

Open3D's tensor raycasting API is a good model for a simple accelerated
geometric backend: rays are shaped `(..., 6)` with origin/direction in the last
dimension, can be image-shaped, and return hit distance. It also has helpers for
pinhole camera rays.

Useful pattern:

```text
image-shaped ray tensors
first-hit distance contract
camera ray creation separate from scene raycast
```

Relevance to Q54:

- good mental model for our `OpticalRaySensorSpec`;
- validates image-shaped depth/segmentation as a natural next step;
- possible optional backend before Embree if dependency policy allows.

Reference:

- https://www.open3d.org/html/python_api/open3d.t.geometry.RaycastingScene.html

### Embree

Embree is an acceleration-kernel library, not a simulator scene model. Its own
site emphasizes replacing the ray tracing operation, dynamic scenes, instancing,
and motion blur. This matches Q54's adapter plan: registry/cache own identity
and buffers; Embree accelerates intersection.

Useful pattern:

```text
our scene/cache -> Embree scene/BVH
our executor contract -> Embree ray queries -> OpticalComputeResult
```

Relevance to Q54:

- strong CPU acceleration candidate for Level 2;
- supports dynamic scene concerns we will need for deformables;
- should not define our registry/result semantics.

Reference:

- https://www.embree.org/

### OptiX

NVIDIA's docs explicitly say OptiX is not itself a renderer; it is a framework
for ray-tracing applications with host-side structures and CUDA programs for
ray generation, intersection, and hit response. This strongly supports keeping
our `OpticalExecutor` contract and treating OptiX as a backend implementation.

Useful pattern:

```text
host acceleration structures + CUDA ray programs
GPU device result path
integration with stream/event lifetime
```

Relevance to Q54:

- good long-term GPU ray tracing path;
- too heavy for Phase A/B because it forces build, SBT, CUDA module, and Q52
  device lifetime decisions.

Reference:

- https://archive.docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_programming_guide.htm

### Mitsuba

Mitsuba's rendering model is scene + sensor + integrator. Rendering is performed
by calling a scene integrator with a sensor, and the result lands in the sensor
film. This is the right mental model for offline/high-fidelity adapters, not
for a realtime RL hot path.

Useful pattern:

```text
scene + sensor + integrator -> film/output
```

Relevance to Q54:

- good offline/reference/differentiable path later;
- not the right first backend for deterministic Phase-A/B contracts.

Reference:

- https://mitsuba2.readthedocs.io/en/latest/src/python_interface/rendering_scene.html

## 3. Capability Levels

### Level 0: First-Hit Range / Material / Instance

Status: implemented.

Current backend:

```text
CpuReferenceOpticalExecutor
```

Inputs:

```text
OpticalSceneSnapshot + OpticalRaySensorSpec
```

Outputs:

```text
range_m
hit_mask
position_world
normal_world
material_id
instance_id
numeric_instance_id
```

Purpose:

- prove scene -> executor -> result lifecycle;
- provide deterministic tests;
- establish channel schema and miss semantics.

Not in scope:

- camera projected depth;
- RGB;
- direct light;
- intensity;
- acceleration.

### Level 1: Image-Shaped Depth / Range / Segmentation Camera

Recommended next capability.

Add a sensor spec or builder for pinhole/orthographic camera rays:

```text
camera intrinsics/extrinsics/resolution
  -> image-shaped ray batch
  -> executor result channels with image shape
```

Outputs:

```text
range_m[H, W]
depth_m[H, W] when optical-axis projection is defined
hit_mask[H, W]
instance_id / numeric_instance_id[H, W]
material_id[H, W]
semantic_id[H, W] when configured
```

Why this next:

- remains geometric and deterministic;
- aligns with Drake/MuJoCo/Habitat-Sim ground-truth modalities;
- exercises image-shaped schema without needing lighting;
- creates a useful RL/debug observation path.

### Level 2: CPU Acceleration

Backends:

```text
Open3D RaycastingScene adapter (optional/simple)
Embree adapter (preferred serious CPU path)
simple in-repo BVH if dependency policy rejects external libs
```

Outputs should match Level 0/1 schemas.

Purpose:

- scale triangle scenes;
- keep exact same `OpticalComputeResult` contract;
- prepare for mesh-heavy robot/world assets.

### Level 3: Direct-Light / Simple RGB Reference

Add a deliberately limited optical model:

```text
first hit
normal_world
material albedo
point/directional lights
optional shadow ray
-> rgb or intensity
```

Outputs:

```text
rgb
intensity
possibly shadow/debug channels
```

Why not earlier:

- requires light semantics;
- requires material response convention;
- requires shadow/occlusion policy;
- starts real optical modeling rather than pure geometry.

### Level 4: Raster / Renderer-Style Camera Backend

Possible paths:

```text
OpenGL/Filament/OGRE-style raster adapter
future rendering/ package bridge
```

Use when the goal is RGB/depth image throughput rather than ray-accurate
LiDAR/segmentation queries.

Risk:

- can leak renderer conventions into Q54 if introduced too early;
- must remain behind `OpticalExecutor`.

### Level 5: GPU / Device Result Path

Backends:

```text
Warp/CUDA custom kernels for simple depth/segmentation
OptiX for triangle-scene ray tracing
```

Requirements:

- `location="device"` result channels;
- ready events/fences;
- Q52 device-consumer completion;
- no Python-level `GpuPublishedFrame` lease inside snapshot;
- packed buffers and acceleration structures in scene/cache.

### Level 6: Offline / High-Fidelity / Volume

Backends:

```text
Mitsuba adapter
path tracing / spectral / differentiable rendering
volume rendering for participating media
fluid/medium optical transport
```

Use for:

- reference images;
- research rendering;
- differentiability;
- validating simpler executors.

Not suitable as the default RL hot path.

## 4. Recommended Q54 Path

Recommended sequence:

```text
0. First-hit range/material/instance reference executor       [done]
1. Image-shaped depth/range/segmentation camera semantics     [done]
2. CPU acceleration adapter: Embree or optional Open3D/BVH
3. Direct-light/simple RGB reference executor
4. Device result path: Warp/CUDA simple kernels, then OptiX
5. Offline/high-fidelity adapter: Mitsuba
6. Fluid/volume optical transport when medium producers exist
```

Rationale:

- Level 1 gives immediately useful camera/depth/segmentation outputs without
  prematurely defining lighting.
- Level 2 makes the geometric executor scalable before adding more optical
  complexity.
- Level 3 introduces light/material semantics only after result schemas and
  geometry acceleration are stable.
- Level 5 should wait until Q52 device lifetime and packed scene buffers are
  exercised on CPU.
- Level 6 should remain an adapter/reference path.

## 5. Immediate Implementation Recommendation

Do not implement direct-light RGB next.

Implement Level 1:

```text
OpticalCameraRaySpec or OpticalPinholeCameraSpec
camera ray builder
image-shaped result schema tests
depth_m vs range_m conversion semantics
segmentation image channels
```

Possible shape:

```text
sensing.OpticalPinholeCameraSpec
  frame_id
  sim_time
  env_idx
  sensor_id
  width
  height
  intrinsics
  X_world_camera
  max_distance
  requested_channels
```

Two implementation options:

1. Build image-shaped rays and reuse `CpuReferenceOpticalExecutor`.
2. Add a camera-specific executor wrapper that calls the ray executor and
   reshapes channels.

Recommendation: start with option 1 plus schema tests. It keeps computation in
the existing reference executor and localizes camera semantics to sensing/spec
builders.

2026-04-30 update: Claude accepted this recommendation and L1 has been
implemented in `sensing/`:

```text
OpticalPinholeCameraSpec
build_pinhole_camera_rays(...)
build_pinhole_camera_image_result(...)
OpticalCameraImageResult
```

The executor remains ray-batch based. `depth_m` is produced by the camera
postprocessor from `range_m` and the camera optical axis.

## 6. Review Questions

1. Is Level 1 image-shaped depth/range/segmentation the right next capability?
2. Should camera ray generation live in `sensing/` as a spec/builder, with the
   executor remaining ray-batch based?
3. Should projected `depth_m` be produced by the camera builder/postprocessor,
   or by a camera-specific executor?
4. Should Open3D be considered as an optional Level-2 backend before Embree, or
   should we go directly to Embree/simple in-repo BVH?
5. Should direct-light RGB wait until CPU acceleration exists?
6. Should raster backends be treated as `OpticalExecutor` adapters or remain in
   `rendering/` until Q54 result contracts need them?
