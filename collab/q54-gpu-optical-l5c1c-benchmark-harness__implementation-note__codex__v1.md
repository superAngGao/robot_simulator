Initiative: q54-gpu-optical-executor
Stage: l5c1c-benchmark-harness-implementation-note
Author: codex
Version: v1
Date: 2026-05-04
Status: implemented
Related Files: benchmarks/bench_optical_device_scene.py, OPEN_QUESTIONS.md, MANIFEST.md
Owner Summary: Added an out-of-pytest benchmark harness for L5C.1c AABB decision-making. It times device-scene update and current derived-layout traversal across configurable ray/triangle cases, including large mesh-heavy presets. After splitting AABB into a true optional variant, H200 xlarge runs show AABB can reduce brute-force traversal time by roughly 20-30% on regular grid triangle scenes, but this remains scene-dependent and should not be confused with BVH-scale acceleration.

# Q54 L5C.1c Benchmark Harness Implementation Note

## 1. Scope

Added:

```text
benchmarks/bench_optical_device_scene.py
```

This is not a pytest test and has no pass/fail performance threshold. It is a
manual decision aid for:

- whether L5C.1c AABB early reject is worth implementing/defaulting;
- when to skip directly to L5C.2 GPU BVH / OptiX evaluation.

## 2. Usage

Smoke:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case smoke --warmup 1 --repeat 2
```

Default five cases:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --warmup 2 --repeat 5
```

Large mesh-heavy cases:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case xlarge

conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case xlarge --use-aabb
```

Custom:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case custom --num-rays 16384 --num-triangles 262144 --warmup 2 --repeat 3
```

CSV output columns:

```text
case,num_rays,num_triangles,
update_ms_mean,update_ms_p50,
execute_ms_mean,execute_ms_p50,
stage_ms_mean
```

`execute_ms` includes ray upload, result allocation, traversal kernel execution,
and synchronization on result readiness. `update_ms` includes device-scene
snapshot update and synchronization on snapshot readiness.

## 3. Cases

```text
few_rays_few_prims:          128 rays,      64 triangles
camera_rays_few_prims:      16384 rays,     64 triangles
camera_rays_mid_prims:      16384 rays,    512 triangles
camera_rays_many_prims:     65536 rays,   2048 triangles
role_filtered_many_prims:   16384 rays,   2048 triangles, about 1/8 visible to depth
large_camera_mid_prims:    262144 rays,   2048 triangles
large_camera_many_prims:   262144 rays,   8192 triangles
role_filtered_large_prims: 262144 rays,   8192 triangles, about 1/8 visible to depth
xlarge_camera_256k_tris:    65536 rays, 262144 triangles
xlarge_mesh_1m_tris:         8192 rays, 1048576 triangles
role_filtered_xlarge_mesh:  65536 rays,1048576 triangles, about 1/32 visible to depth
```

The benchmark uses deterministic world-static grid triangle meshes and
downward camera-like rays. It is not random geometry and not a robot mesh.
The role-filtered case splits visible/hidden triangles into separate instances
so the existing per-primitive role-mask branch is exercised.

For xlarge cases, the default warmup/repeat drops to 1/3 based on the
`num_rays * num_triangles` work estimate. Explicit `--warmup` and `--repeat`
still override this.

## 4. Initial H200 Baseline Before AABB Variant

Command:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --warmup 2 --repeat 5
```

Result:

```text
case,num_rays,num_triangles,update_ms_mean,update_ms_p50,execute_ms_mean,execute_ms_p50,stage_ms_mean
few_rays_few_prims,128,64,0.1091,0.1080,0.2781,0.2808,0.0000
camera_rays_few_prims,16384,64,0.1070,0.1064,0.4207,0.4091,0.0000
camera_rays_mid_prims,16384,512,0.1165,0.1143,0.5975,0.6001,0.0000
camera_rays_many_prims,65536,2048,0.1190,0.1195,2.3071,2.3046,0.0000
role_filtered_many_prims,16384,2048,0.1116,0.1139,0.6709,0.6646,0.0000
```

Interpretation:

- update cost is currently low and nearly flat for these sizes;
- traversal/executor cost dominates large camera/mesh cases;
- role filtering can materially reduce traversal math when many primitives are
  invisible, even though the loop still scans all primitives;
- AABB is worth implementing as a measurable variant, but not defaulting until
  we have with-AABB numbers.

## 5. Clean AABB Variant Xlarge Baseline

After splitting AABB into a true optional path:

```text
use_aabb=False:
  no AABB buffers allocated;
  no AABB parameters passed to traversal;
  no runtime AABB branch in the traversal kernel.

use_aabb=True:
  triangle_aabb_min/max allocated and filled;
  AABB traversal kernel used.
```

Earlier H200 xlarge results before increasing xlarge to million-triangle
presets:

```text
case,use_aabb,num_rays,num_triangles,update_ms_mean,update_ms_p50,execute_ms_mean,execute_ms_p50,stage_ms_mean
xlarge_mesh_256k_tris,0,16384,262144,2.4068,2.4094,307.7941,308.0229,0.0000
xlarge_mesh_256k_tris,1,16384,262144,2.4238,2.4259,235.6695,235.6132,0.0000

xlarge_mesh_64k_tris,0,65536,65536,4.6791,4.6781,226.7578,226.6779,0.0000
xlarge_mesh_64k_tris,1,65536,65536,3.9206,4.6738,156.1677,172.1525,0.0000
```

After increasing xlarge to million-triangle mesh pressure:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case xlarge_mesh_1m_tris --warmup 1 --repeat 1

conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case xlarge_mesh_1m_tris --warmup 1 --repeat 1 --use-aabb
```

Result:

```text
case,use_aabb,num_rays,num_triangles,update_ms_mean,update_ms_p50,execute_ms_mean,execute_ms_p50,stage_ms_mean
xlarge_mesh_1m_tris,0,8192,1048576,2.4025,2.4025,1195.4274,1195.4274,0.0000
xlarge_mesh_1m_tris,1,8192,1048576,2.4825,2.4825,919.2924,919.2924,0.0000
```

Interpretation:

- On these regular grid scenes, AABB reduces traversal time by roughly 20-30%.
- This is a useful constant-factor improvement but does not change the
  brute-force `O(num_rays * num_triangles)` scaling.
- The geometry is still artificial; add random/clustered/robot-like mesh cases
  before making broad default-policy claims.
- The previous xlarge presets were still too small on triangle count. The
  harness now includes 262k-triangle camera pressure and 1M-triangle mesh-heavy
  cases, with vectorized deterministic mesh generation to avoid Python setup
  dominating large runs.

## 6. Deferred

## 6. Robot-like Scene Extension

Added shared robot-like scene generation in:

```text
benchmarks/robot_optical_scene.py
examples/optical_robot_scene_preview.py
```

Benchmark cases:

```text
robot_proxy_pose:    16384 rays,   1104 triangles, 1 body-bound quadruped
robot_dense_single:  65536 rays, 161280 triangles, 1 body-bound quadruped
robot_dense_pack:    65536 rays, 645120 triangles, 4 body-bound quadrupeds
robot_ego_camera:    65536 rays, 161280 triangles, 1 body-bound quadruped, close ego view
```

The robot scene is deterministic and generated in-repo. It uses one torso plus
four leg chains, with 13 body-bound mesh instances per robot. It is not a
URDF visual asset, but it exercises the important layout/performance features
that regular grids miss: clustered link geometry, self-occlusion, multiple
body transforms, source-order tie-breaks across instances, and camera-like
rays.

Initial H200 robot dense single result:

```text
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --warmup 1 --repeat 3

conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --warmup 1 --repeat 3 --use-aabb
```

Result:

```text
case,use_aabb,num_rays,num_triangles,update_ms_mean,update_ms_p50,execute_ms_mean,execute_ms_p50,stage_ms_mean
robot_dense_single,0,65536,161280,2.4435,2.4516,331.5016,331.2295,0.0000
robot_dense_single,1,65536,161280,2.4408,2.4315,267.0543,267.9024,0.0000
```

Interpretation:

- On this robot-like clustered scene, AABB reduces traversal by about 19% in
  the repeat=3 short run.
- This is still a brute-force traversal; the scene is useful for deciding
  whether per-triangle AABB is worth keeping, but it strengthens the case for
  BVH/OptiX rather than replacing it.
- Preview images can be generated with:

```text
python examples/optical_robot_scene_preview.py --width 480 --height 320 --out out/optical_robot_scene
```

## 7. Deferred

The benchmark does not yet cover:

- random or clustered triangle distributions;
- body-bound rotating mesh multi-frame timing;
- GPU BVH / OptiX traversal.

Those should be added before making a default acceleration decision.
