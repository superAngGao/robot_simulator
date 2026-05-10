# Q54 Optical Pipeline Lab E2 Shadow Optimization Plan

Date: 2026-05-10
Author: Codex
Status: measured-plan

## Context

E1 split `render_execute_ms` from `pack_rgb8_ms` and added render-profile
overhead accounting. The first 1080p shadow RGB8 profile showed the render-side
budget is now dominated by executor render, not RGB8 delivery:

```text
raygen_kernel    ~= 0.17-0.18 ms
first_hit_kernel ~= 1.96-1.97 ms
shade_kernel     ~= 3.16-3.20 ms
pack_rgb8        ~= 0.08-0.12 ms
```

That makes shadow/shade the next sensible optimization target.

## Measurement First

Before changing kernels, run a same-device E1 profile comparison:

```text
1080p shadow,    readback=rgb8, torch_async ring2, render-profile on
1080p no-shadow, readback=rgb8, torch_async ring2, render-profile on
1080p shadow,    readback=none, render-profile on
1080p no-shadow, readback=none, render-profile on
```

Use an idle GPU when possible. If all GPUs are shared, mark the run as
diagnostic only and do not update baseline numbers from it.

The key question is whether the delta is mostly:

```text
shade_kernel_ms only
first_hit_kernel_ms + shade_kernel_ms interaction
allocation/profile overhead
readback/delivery noise
```

## E2.0 Measurement

Environment:

```text
device: cuda:1 / NVIDIA H200
resolution: 1920x1080
frames: 20
warmup_renders: 5
steady window: frames 1-18, excluding first and final async-drain frame
out: out/optical_pipeline_lab/e2_shadow_profile_gpu1/
```

RGB8 async preview path:

```text
shadow_rgb8_async:
  frame_total_ms        ~= 7.691
  render_execute_ms     ~= 7.128
  raygen_kernel_ms      ~= 0.783
  first_hit_kernel_ms   ~= 2.085
  shade_kernel_ms       ~= 3.379
  pack_rgb8_ms          ~= 0.091
  readback_host_ms      ~= 2.313

no_shadow_rgb8_async:
  frame_total_ms        ~= 4.525
  render_execute_ms     ~= 3.952
  raygen_kernel_ms      ~= 0.709
  first_hit_kernel_ms   ~= 2.091
  shade_kernel_ms       ~= 0.255
  pack_rgb8_ms          ~= 0.090
  readback_host_ms      ~= 2.345

delta shadow - no_shadow:
  frame_total_ms        ~= +3.166
  render_execute_ms     ~= +3.176
  shade_kernel_ms       ~= +3.124
  first_hit_kernel_ms   ~= -0.006
  readback_host_ms      ~= -0.032
```

Readback-none render path:

```text
shadow_none_repeat:
  frame_total_ms        ~= 6.252
  render_execute_ms     ~= 6.102
  raygen_kernel_ms      ~= 0.292
  first_hit_kernel_ms   ~= 2.037
  shade_kernel_ms       ~= 3.371

no_shadow_none:
  frame_total_ms        ~= 3.041
  render_execute_ms     ~= 2.894
  raygen_kernel_ms      ~= 0.218
  first_hit_kernel_ms   ~= 2.038
  shade_kernel_ms       ~= 0.254

delta shadow - no_shadow:
  frame_total_ms        ~= +3.211
  render_execute_ms     ~= +3.208
  shade_kernel_ms       ~= +3.117
  first_hit_kernel_ms   ~= -0.001
```

The first `shadow_none` run had allocation/raygen spikes and is treated as an
outlier. The repeat is stable and agrees with the RGB8 async preview path.

Conclusion: the measured hard-shadow cost is about 3.1 ms at 1080p on this Go2
scene, and the delta is almost entirely inside `shade_kernel_ms`. First-hit,
RGB8 pack, and readback do not explain the shadow cost.

## Current Shadow Hot Path

The direct-light executor currently shades every hit ray in one kernel. For
each positive `n_dot_l` light contribution, the shadow path calls an inline
BVH any-hit traversal:

```text
_device_scene_direct_light_kernel
  -> _is_occluded_for_ray
```

Important current behavior:

- shadow traversal uses per-ray local stack arrays;
- shadow diagnostics are updated through atomics;
- plane occlusion is checked after BVH traversal when still unoccluded;
- RGB preview still allocates and writes `intensity`, then filters it out;
- point lights recompute distance for `shadow_max_distance`, although the
  direction normalization path has already computed the same distance.

## Candidate Slices

Recommended order:

1. Profile-only comparison: complete. Shadow cost is ~3.1 ms and is localized
   to `shade_kernel_ms`.
2. Low-risk cleanup: avoid duplicate point-light distance computation in the
   shade kernel. This may be neutral for Go2 if the scene is directional-light
   dominated, but it is simple and safe.
3. Output-profile cleanup: avoid allocating/writing `intensity` for RGB preview
   when no consumer requests it.
4. Shadow diagnostic policy: make per-frame shadow diagnostic atomics optional
   or reduce their frequency if profile shows they matter.
5. Shadow traversal experiment: benchmark a stack-only any-hit traversal that
   drops the per-node `stack_t` local array. Keep this behind an experiment flag
   until measured.
6. Kernel specialization: only split no-shadow and shadow shade kernels if the
   no-shadow profile shows register pressure or branch cost from the shared
   kernel is visible.

## E2.1 Experiment: Stack-Only Shadow Any-Hit

Hypothesis:

```text
The per-shadow-ray `stack_t` local array may not be the dominant latency source.
```

Why test it anyway:

- `stack_t` consumes 32 float32 local slots per shadow ray;
- shadow any-hit only needs visibility, not nearest-hit ordering;
- every pushed child has already passed an AABB interval test with
  `max_distance`, so the later `node_t <= max_distance` pop check should be
  redundant for correctness;
- removing it is a small, reversible experiment that directly touches the
  measured `shade_kernel_ms` hot path.

Acceptance:

```text
correctness:
  shadow_stack_overflow_count remains 0
  visual output is not inspected in this slice, but hit/overflow diagnostics
  must stay sane

performance:
  compare against E2.0 shadow_rgb8_async steady frames
  baseline shade_kernel_ms ~= 3.379
  useful win threshold: >= 5% shade_kernel_ms reduction
```

If the result is neutral, treat stack traversal itself rather than `stack_t`
storage as the likely source, and move to diagnostics policy or a larger
shadow traversal specialization experiment.

Result:

```text
baseline shadow_rgb8_async, steady frames 1-18:
  frame_total_ms        ~= 7.691
  render_execute_ms     ~= 7.128
  shade_kernel_ms       ~= 3.379
  first_hit_kernel_ms   ~= 2.085
  readback_host_ms      ~= 2.313
  shadow_overflow       = 0

stack_only shadow_rgb8_async, steady frames 1-18:
  frame_total_ms        ~= 7.033
  render_execute_ms     ~= 6.456
  shade_kernel_ms       ~= 2.779
  first_hit_kernel_ms   ~= 2.091
  readback_host_ms      ~= 2.312
  shadow_overflow       = 0

delta stack_only - baseline:
  frame_total_ms        ~= -0.658
  render_execute_ms     ~= -0.672
  shade_kernel_ms       ~= -0.600
  shade speedup         ~= 17.8%
```

A repeat run had slower system-level readback/BVH/warmup timings, but
`shade_kernel_ms` stayed close at ~= 2.816 ms. That repeat supports the local
kernel conclusion while not being suitable as an overall throughput baseline.

Interpretation: `stack_t` was not the whole 3.1 ms shadow cost, but it was a real
chunk of it. Removing it recovers about 0.6 ms from `shade_kernel_ms`; the
remaining ~2.5-2.8 ms is still the shadow traversal itself.

## Non-Goals For This Slice

- Do not start path tracing.
- Do not redesign the public simulator-facing API.
- Do not introduce a separate shadow-ray queue until the inline any-hit path is
  proven to be the bottleneck under E1 profile data.
- Do not optimize from profile-on readback copy timings; profiling adds
  synchronization and those rows are diagnostic proportions, not throughput
  baselines.

## Expected Review Questions

1. Is `shade_kernel_ms` the right first target after RGB8 delivery?
2. Should `intensity` be retained for `RGB_PREVIEW`, or is it safe to skip?
3. Are shadow diagnostics required for normal preview runs, or only debug/full
   output profiles?
4. Is stack-only any-hit traversal worth benchmarking before a bigger kernel
   split?
5. Should no-shadow specialization wait until after output-profile cleanup?
