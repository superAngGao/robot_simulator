# Q54 GPU Optical L5C.3b GPU Preview Implementation Note

Date: 2026-05-05

## Scope

Added a GPU MuJoCo Menagerie preview example:

```bash
examples/mujoco_menagerie_gpu_preview.py
```

The script imports the existing MuJoCo Menagerie visual mesh scene, then renders
with:

- `DeviceOpticalSceneCache`
- `build_device_bvh_from_snapshot`
- `GpuDeviceBvhDirectLightOpticalExecutor(shadows=True)`
- `stage_optical_compute_result_to_host`
- existing PNG preview writers

It writes RGB, projected depth, numeric instance segmentation, and a panel PNG.
The panel title now says `GPU direct-light RGB`.

## Commands Run

Smoke:

```bash
conda run -n env_tilelang_20260119 python examples/mujoco_menagerie_gpu_preview.py \
  --width 320 --height 220 \
  --out out/menagerie_go2_gpu_preview_smoke \
  --fail-on-overflow
```

Full front view:

```bash
conda run -n env_tilelang_20260119 python examples/mujoco_menagerie_gpu_preview.py \
  --width 960 --height 640 \
  --out out/menagerie_go2_gpu_preview_front \
  --fail-on-overflow
```

Full side view:

```bash
conda run -n env_tilelang_20260119 python examples/mujoco_menagerie_gpu_preview.py \
  --width 960 --height 640 \
  --view side \
  --out out/menagerie_go2_gpu_preview_side \
  --fail-on-overflow
```

## Results

Scene:

- MuJoCo Menagerie Unitree Go2 visual mesh
- 33 visual geoms
- 398,432 triangles

Front 960x640:

- `primary_overflow=0`
- `primary_max_stack=17`
- `shadow_overflow=0`
- `shadow_max_stack=13`
- elapsed wall time: about 23.3 s
- outputs:
  - `out/menagerie_go2_gpu_preview_front/rgb.png`
  - `out/menagerie_go2_gpu_preview_front/depth_m.png`
  - `out/menagerie_go2_gpu_preview_front/numeric_instance_id.png`
  - `out/menagerie_go2_gpu_preview_front/panel.png`

Side 960x640:

- `primary_overflow=0`
- `primary_max_stack=15`
- `shadow_overflow=0`
- `shadow_max_stack=13`
- elapsed wall time: about 22.9 s
- outputs:
  - `out/menagerie_go2_gpu_preview_side/rgb.png`
  - `out/menagerie_go2_gpu_preview_side/depth_m.png`
  - `out/menagerie_go2_gpu_preview_side/numeric_instance_id.png`
  - `out/menagerie_go2_gpu_preview_side/panel.png`

Both panels were inspected visually. RGB, depth, and segmentation are nonblank,
and hard shadows are visible.

## Notes

- This is direct lighting with hard shadows, not path tracing.
- Path tracing / stray-light transport remains deferred to a separate executor
  family.
- Current elapsed time includes mesh import, device scene upload, BVH build,
  rendering, staging, and PNG writing. It is not a pure kernel timing.
- The script defaults to suppressing Warp logs so stdout stays readable; use
  `--verbose-warp` for module-load debugging.
