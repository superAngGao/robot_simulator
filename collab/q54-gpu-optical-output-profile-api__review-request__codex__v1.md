Initiative: q54-gpu-optical-output-profile-api
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-07
Status: reviewed-followup
Related Files: optics/execution.py, optics/warp_execution.py, optics/device.py, examples/mujoco_menagerie_gpu_preview.py, sensing/optical.py
Owner Summary: V1 video benchmark cleanup proved that selected host readback is useful, but it also exposed a deeper issue: RGB/video fast paths should not be implemented by silently dropping channels after a full GPU result has already been computed. We need an explicit output-profile contract so callers can request full geometry/debug results, RGB preview results, or render-only timing without ambiguity.

# Q54 GPU Optical Output Profile API Review Request

## Background

The current GPU direct-light executor returns a full per-ray result contract:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
rgb
intensity
bvh_stack_overflow_count
bvh_max_stack_depth
shadow_stack_overflow_count
shadow_max_stack_depth
```

The V1 video benchmark added:

```text
--video-readback full|rgb|none
stage_optical_channels(result, channels)
```

This helps host readback:

```text
full readback:  ~32-35 ms at 960x640
rgb readback:   ~5.9 ms at 960x640
none:           render-only benchmark
```

But selected readback is only half the story. In `rgb` or `none` mode, the
current executor may still allocate, compute, and write channels such as
`position_world`, `normal_world`, and `range_m`. That means:

```text
we reduce host transfer cost,
but we do not necessarily reduce GPU kernel work or device memory traffic.
```

If we optimize kernel output by silently not writing those buffers while the
public executor still claims to return a full result, callers will get confusing
`KeyError`s or, worse, stale/uninitialized channels.

Therefore the fast path needs an explicit API contract.

## Goal

Introduce an explicit output-profile concept for optical executors.

The output profile should answer:

```text
Which channels are guaranteed to exist?
Which device buffers must be allocated?
Which channels must the kernel compute/write?
Which downstream consumers are allowed to use the result?
```

The first version should be small and profile-based, not a free-form channel
list.

## Non-Goals

For the first design:

```text
do not expose arbitrary per-channel masks to public users
do not rewrite traversal to CUDA/OptiX
do not implement path tracing profiles yet
do not remove the existing full-result path
do not change CPU reference semantics unless necessary
```

## Proposed Profiles

Use a small enum-like profile set:

```text
geometry_full
direct_light_full
rgb_preview
render_only
```

### geometry_full

Purpose:

```text
first-hit geometry query
sensor/depth/parity/debug
```

Required channels:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
bvh_stack_overflow_count
bvh_max_stack_depth
```

No shading channels:

```text
rgb
intensity
shadow_stack_overflow_count
shadow_max_stack_depth
```

### direct_light_full

Purpose:

```text
full direct-light semantic result
CPU/GPU parity
debug image generation
depth/range/segmentation/RGB combined preview
```

Required channels:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
rgb
intensity
bvh_stack_overflow_count
bvh_max_stack_depth
shadow_stack_overflow_count
shadow_max_stack_depth
```

This should remain the default for existing direct-light executor calls to
avoid breaking tests and debug behavior.

### rgb_preview

Purpose:

```text
fast RGB preview/video path
```

Required channels:

```text
rgb
bvh_stack_overflow_count
bvh_max_stack_depth
shadow_stack_overflow_count
shadow_max_stack_depth
```

Not guaranteed:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
intensity
```

Kernel/buffer implication:

```text
do not allocate/write large geometry output buffers unless needed internally
do not write intensity unless explicitly needed
keep diagnostic scalar outputs
```

Important nuance:

The direct-light shader still needs hit position/normal/material internally to
shade. The profile only says those intermediate values are not materialized as
public per-ray output channels.

### render_only

Purpose:

```text
kernel timing / smoke benchmark
possibly future GPU-only downstream consumers
```

Required channels:

```text
none, or diagnostics only if cheap and explicitly documented
```

Result contract:

```text
ready_event is meaningful
channels may be empty
host readback is not implied
```

Open detail:

Should `render_only` still return overflow diagnostics? Returning diagnostics
requires some output buffers and optional host readback, but having them is very
useful for safety. V1 benchmark currently treats diagnostics as NaN in
`--video-readback none`.

## API Surface

Short-term executor API:

```python
executor.execute(snapshot, bvh, rays, output_profile="direct_light_full")
```

For first-hit executor:

```python
first_hit_executor.execute(snapshot, bvh, rays, output_profile="geometry_full")
```

For direct-light executor:

```python
direct_executor.execute(snapshot, bvh, rays, output_profile="direct_light_full")
direct_executor.execute(snapshot, bvh, rays, output_profile="rgb_preview")
direct_executor.execute(snapshot, bvh, rays, output_profile="render_only")
```

Potential type shape:

```python
OpticalOutputProfile = Literal[
    "geometry_full",
    "direct_light_full",
    "rgb_preview",
    "render_only",
]
```

or a small `Enum`.

Recommendation:

```text
Use string Literal first for low churn and CLI friendliness.
Consider Enum later when the formal render session API is introduced.
```

## `OpticalComputeResult` Contract

Current behavior:

```python
result.channel("position_world")
```

raises `KeyError` if the channel does not exist.

Recommendation:

```text
Keep this behavior.
Add output_profile metadata to OpticalComputeResult so missing channels are
understandable by contract rather than surprising.
```

Proposed field:

```python
output_profile: str = "direct_light_full"
```

Documentation:

```text
Callers must check `result.output_profile` or `result.channels` before assuming
that optional channels exist.
```

Potential helper:

```python
result.has_channel("rgb") -> bool
```

This helper is optional but may make profile-aware consumers cleaner.

## Buffer / Kernel Design Implications

The output profile must be passed before buffer allocation.

Current full path likely allocates:

```text
hit_mask[N]
range_m[N]
position_world[N,3]
normal_world[N,3]
numeric_instance_id[N]
rgb[N,3]
intensity[N]
diagnostic scalars
```

For `rgb_preview`, preferred allocation:

```text
rgb[N,3]
diagnostic scalars
temporary/intermediate hit fields only if required by kernel design
```

For `render_only`, preferred allocation:

```text
diagnostic scalars or no public channels
```

Implementation detail:

```text
The first implementation may still use some internal temporary buffers if that
keeps parity risk low. The public contract should nevertheless be profile-based,
so later fused kernels can remove temporaries without changing API.
```

## Compatibility Strategy

Phase 1:

```text
Add profile metadata and validation.
Default direct-light profile remains direct_light_full.
Existing tests should continue to pass.
Video benchmark can request rgb_preview/render_only once implemented.
```

Phase 2:

```text
Avoid public channel allocation/write for rgb_preview.
Keep direct_light_full unchanged.
Add parity tests comparing rgb_preview rgb against direct_light_full rgb.
```

Phase 3:

```text
Add camera-specific GPU pinhole raygen behind the same profile contract.
Then fuse raygen/trace/shade or add CUDA/OptiX backend without changing the
public output-profile semantics.

The generic ray-batch path remains available for LiDAR, arbitrary ray queries,
CPU/GPU parity, and tests that need explicit origins/directions.
```

## Testing Plan

Unit/API tests:

```text
direct_light_full exposes current full channel set
rgb_preview exposes rgb + diagnostics and does not expose position/range/normal
render_only exposes empty or diagnostics-only channels as documented
result.channel(missing) raises KeyError
result.output_profile matches requested profile
```

GPU parity tests:

```text
rgb_preview rgb matches direct_light_full rgb within tolerance
rgb_preview overflow diagnostics match direct_light_full diagnostics
render_only launches and signals ready_event
render_only rejects/handles fail-on-overflow policy explicitly
```

Benchmark tests:

```text
960x640 static Go2:
  direct_light_full render/readback baseline
  rgb_preview render + rgb readback
  render_only event wait
```

## Open Questions For Claude

1. Is a small profile set (`geometry_full`, `direct_light_full`,
   `rgb_preview`, `render_only`) the right public surface, or should we expose a
   lower-level channel mask sooner?

2. Should `output_profile` be a string `Literal` now, or should we introduce an
   enum/dataclass immediately?

3. Should `render_only` return diagnostic scalar channels, or should it return
   no channels and require a separate debug mode for diagnostics?

4. For `rgb_preview`, should `hit_mask` be included? It is useful for debugging
   and background handling, but it is an additional per-ray output. Current RGB
   shader already writes background RGB, so the video path does not require it.

5. Should `intensity` be part of `rgb_preview`? It is cheap compared with
   position/normal, but not needed for RGB video.

6. Should `OpticalComputeResult` gain `has_channel(...)`, or is checking
   `name in result.channels` sufficient?

7. Should CPU direct-light executor also support `output_profile`, or should the
   first implementation restrict profiles to GPU direct-light only?

8. How strict should validation be when a consumer asks
   `build_pinhole_camera_image_result(...)` to consume an `rgb_preview` result
   without range/depth channels?

## Codex Recommendation

Use profile-based API now, with `direct_light_full` as the default.

Implement `rgb_preview` first for the GPU direct-light executor:

```text
public channels:
  rgb + diagnostics

no public geometry channels:
  no range_m, position_world, normal_world, numeric_instance_id
```

Keep `render_only` as a benchmark-only profile until diagnostic semantics are
settled.

Do not expose free-form channel masks yet. The current need is not arbitrary
customization; it is three clear use cases:

```text
full semantic result
fast RGB preview/video
render-only timing
```

## Claude Review Follow-Up

Claude reviewed this design on 2026-05-07. Codex accepts the main changes.

Accepted decisions:

```text
1. Use OpticalOutputProfile Enum instead of string Literal.
2. Encode guaranteed_channels on the Enum, not only in docs.
3. Add is_superset_of(...) so consumers can ask profile inclusion questions.
4. Add output_profile to OpticalComputeResult.
5. Keep result.has_channel(name) as runtime channel-dict membership.
   Static guarantees should be checked through result.output_profile or
   profile.guaranteed_channels, not hidden behind has_channel.
6. render_only keeps diagnostic scalar channels, because overflow diagnostics
   are safety signals and cheap.
7. rgb_preview includes hit_mask + rgb + diagnostics.
   It does not include intensity.
8. Executors declare supported_profiles and validate at execute() entry.
9. CPU direct-light should support profiles too, initially by filtering output
   channels, so CPU/GPU parity can cover rgb_preview.
10. build_pinhole_camera_image_result should fail fast when required channels
    are missing for the requested image construction.
11. OpticalComputeResult.ready_event docstring must clarify:
    ready_event means all channels present in result.channels / guaranteed by
    the requested output profile have been written to device buffers; host
    readback is the caller's responsibility.
```

Implementation order:

```text
1. Add OpticalOutputProfile Enum and channel guarantee helpers.
2. Add output_profile + has_channel to OpticalComputeResult.
3. Preserve direct_light_full as default everywhere.
4. Add CPU direct-light profile filtering.
5. Add GPU direct-light profile validation and public-channel filtering.
   First implementation may still compute full internals; later kernel work can
   use the same contract to skip allocations/writes.
6. Add image-builder fail-fast checks.
7. Add tests for profile contracts and profile-filtered CPU/GPU parity.
```
