# Q54 Optical Pipeline Lab — E0/E1 RenderSession + Render Timing Plan

Date: 2026-05-10
Author: Codex
Status: review-request

## Context

D2 RGB8 pack + async readback is merged at:

```text
cc46f8c Add optical pipeline lab and RGB8 readback
```

Current GPU1 1080p shadow RGB8 async ring2 result:

```text
all10:
  frame_mean       ~= 6.71 ms
  render+pack_mean ~= 6.04 ms
  readback_copy    ~= 2.33 ms

steady frames 1-8:
  frame_mean       ~= 6.44 ms
  readback_copy    ~= 2.33 ms
```

Conclusion from D2 review: readback is no longer the dominant bottleneck. The
next useful target is the roughly 6 ms render-side budget.

## Problem

`tools/optical_pipeline_lab/go2_backend.py` currently mixes three concerns:

1. Go2 render lifecycle:
   - `wp.init`
   - device/stream
   - scene import
   - static GPU frame
   - `DeviceOpticalSceneCache`
   - snapshot
   - BVH
   - executor
2. Per-frame render loop:
   - camera/ray construction
   - `execute_camera()` or `execute()`
   - optional RGB8 pack
   - sync vs torch_async readback
   - frame CSV row emission
3. Render profiling:
   - optional `render_profile` executor list
   - existing `render_*_ms` CSV columns
   - currently coarse `render_execute_ms`

This is acceptable for C/D lab work, but Stage I needs a clearer session/workspace
boundary before we optimize render scheduling or buffer reuse.

## Proposed Slice

Do this in small steps. Avoid a broad production runtime rewrite.

### E0.1 — Introduce Go2RenderSession

Add a small session object inside `tools/optical_pipeline_lab/go2_backend.py`
or a new adjacent module if review prefers.

Candidate shape:

```python
@dataclass
class Go2RenderSession:
    scene: object
    device: object
    stream: object
    gpu_frame: GpuPublishedFrame
    cache: DeviceOpticalSceneCache
    snapshot: object
    bvh: object
    executor: GpuDeviceBvhDirectLightOpticalExecutor

    @classmethod
    def create(cls, args: argparse.Namespace, timings: TimingRecorder) -> "Go2RenderSession":
        ...

    def execute_frame(...):
        ...

    def pack_rgb8(...):
        ...
```

Rules:

- Preserve current behavior.
- Keep the existing CLI and lab runner surface unchanged.
- Keep sync and `torch_async` readback behavior unchanged.
- Do not move async ring ownership into the session yet.
- Do not introduce a production `optics/` runtime abstraction yet.

Rationale:

This creates the lifecycle boundary Stage I needs without committing to the
final production API too early.

### E0.2 — Share The Render Half Of Sync/Async Loops

Extract the duplicated per-frame render preparation used by both sync and
`torch_async` paths:

```python
def _render_video_frame(session, args, frame_index, ray_cache):
    ...
    return _RenderedVideoFrame(
        camera=...,
        result=...,
        camera_rays_ms=...,
        render_execute_ms=...,
        pack_rgb8_ms=...,
        render_profile_row=...,
    )
```

Sync and async paths should still own their respective readback completion and
CSV row emission, but the render half should become one call.

Rationale:

Future scheduling work needs render as a single explicit unit. Today the sync
and async loops duplicate too much logic to safely tune one path.

### E1.1 — Split RGB8 Pack Timing Out Of Render Execute

Current D2 timing folds RGB8 pack into `render_execute_ms`:

```text
render+pack_mean ~= 6.04 ms
```

Add a stable CSV field:

```text
pack_rgb8_ms
```

Semantics:

- `render_execute_ms`: executor render only, including the existing
  `wp.synchronize_event(result.ready_event)`.
- `pack_rgb8_ms`: RGB8 pack launch + completion wait.
- `pack_rgb8_ms = NaN` unless `readback=rgb8`.
- `frame_total_ms` behavior unchanged.

This lets us answer whether RGB8 pack is visible inside the remaining ~6 ms
budget.

### E1.2 — Fill render_overhead_ms When render_profile Is Enabled

The design document already defines:

```text
render_overhead_ms =
  render_execute_ms - sum(recorded render kernel/profile subphases)
```

Implement this only when `--render-profile` is enabled and profile rows exist.

Rules:

- Keep `render_overhead_ms = NaN` when profiling is disabled.
- Clamp only if needed for tiny negative timing noise, or leave the raw value
  and let review decide.
- Do not treat profile-on throughput as the real performance number, because
  profiling inserts extra synchronization.

### E1.3 — GPU1 Measurement

After E0.1/E1.1, run:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_e0e1_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2
```

After E1.2, run the same case with:

```text
--render-profile
```

Expected outputs:

```text
frame_total_ms
render_execute_ms
pack_rgb8_ms
readback_host_ms
render_raygen_kernel_ms
render_first_hit_kernel_ms
render_shade_kernel_ms
render_overhead_ms
```

## Non-goals

- No OptiX/CUDA render rewrite.
- No multi-stream render scheduling yet.
- No dynamic geometry/refit work.
- No replacement of the lab runner API.
- No async ring redesign unless extraction exposes a correctness issue.
- No new production `optics.RenderSession` until the lab boundary proves useful.

## Acceptance Criteria

- Existing CLI behavior remains compatible.
- `examples/mujoco_menagerie_gpu_preview.py` still works as a thin wrapper.
- Unit tests pass:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
```

- Formatting/lint pass:

```text
conda run -n env_tilelang_20260119 ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
```

- GPU smoke passes on GPU1 with RGB8 async ring2.
- CSV includes `pack_rgb8_ms`.
- With `--render-profile`, CSV fills `render_overhead_ms`.
- D2 baseline remains within expected noise unless profiling is enabled.

## Review Questions

1. Should `Go2RenderSession` live inside `go2_backend.py` for this slice, or in
   a separate `go2_session.py` module?
2. Is `pack_rgb8_ms` the right schema name, or should it be more generic
   (`delivery_pack_ms`, `output_pack_ms`) for future payloads?
3. Should `render_execute_ms` exclude RGB8 pack as proposed, or should we keep
   `render_execute_ms` as render+pack and add `render_core_ms`?
4. Is it acceptable that `render_overhead_ms` is only meaningful with
   `--render-profile` enabled?
5. Should async ring ownership stay outside `Go2RenderSession` for now?

## Recommended First Implementation

Start with E0.1 + E1.1 only:

```text
Go2RenderSession skeleton
shared render-frame helper
pack_rgb8_ms CSV field
no render_overhead_ms yet
GPU1 smoke
```

Then do E1.2 as the second small change after review confirms the timing
semantics.
