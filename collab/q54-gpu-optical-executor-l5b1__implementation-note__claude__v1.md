Initiative: q54-gpu-optical-executor
Stage: l5b1-implementation-note
Author: claude
Version: v1
Date: 2026-05-03
Status: implemented
Related Files: optics/gpu_runtime.py, tests/gpu/test_optical_gpu_runtime.py, MANIFEST.md, OPEN_QUESTIONS.md

# Q54 L5B.1 Body-Bound GPU Optical Runtime — Claude Implementation Note

## 1. Scope

L5B.1 extends `execute_optical_on_gpu_published_frame` to support
`OpticalInstanceSpec(body_index=i)` by host-staging the selected env's body
transforms from `GpuPublishedFrame.x_world_R_wp` / `x_world_r_wp` before
GPU primitive upload.

The old `_require_world_static_registry` guard and the corresponding
`test_execute_optical_on_gpu_published_frame_rejects_body_bound_registry` test
are both removed; body-bound registries are now valid inputs.

## 2. 关键思考

### 2.1 Host-staged transforms vs. device-resident scene cache

**Decision**: stage body transforms to host NumPy, build a `CpuPublishedFrame`
with those transforms, then call the existing `OpticalSceneCache` path.

**Why this is the right transitional step**: the existing
`OpticalSceneCache.snapshot_from_frame_inputs` already handles body-bound
instance geometry correctly — it reads `X_world[body_index]` and applies the
transform to each primitive. Reusing that path means L5B.1 adds zero new
geometry-transform logic; it only adds the host-staging bridge.

**Why this is not L5C**: every call re-packs world-space primitives from
scratch. For many envs or high-resolution cameras this is the bottleneck.
L5C should move primitive packing to device kernels that read `x_world_*`
directly, eliminating the host round-trip.

**Alternative considered**: add a new device-side transform kernel now.
Rejected: premature. The host-staged path validates the full registry/frame/
result semantic contract first. Building a device scene cache on top of a
validated contract is safer than building both at once.

### 2.2 Synchronizing the frame ready event before host staging

`_host_stage_body_transforms_if_needed` calls
`_synchronize_frame_ready_event(frame.ready_event)` before calling
`.numpy()` on the Warp arrays.

This is necessary because `GpuPublishedFrame` is produced asynchronously.
Without the sync, `.numpy()` could read partially-written transform data.
The sync is conditional on `max_body_index is not None` — world-static
registries skip it entirely, preserving the L5B.0 behavior.

**Risk**: if the caller has already synchronized the frame (as the body-bound
tests do with `wp.synchronize_event(frame.ready_event)`), the sync is a
no-op. Double-sync is safe.

### 2.3 Removing the `_require_world_static_registry` guard

The L5B.0 guard was a deliberate "fail loudly rather than silently wrong"
measure. L5B.1 replaces it with correct behavior, so the guard is removed.

The corresponding test `test_execute_optical_on_gpu_published_frame_rejects_body_bound_registry`
is also removed — it tested a constraint that no longer exists.

**Risk of removing a test**: the new
`test_execute_optical_on_gpu_published_frame_supports_body_bound_registry`
provides stronger coverage: it verifies the correct numeric result
(`range_m = 2.0 - body_z`, `position_world[0] = [0, 0, body_z]`) rather
than just that an exception is raised.

### 2.4 Shape validation for staged transform arrays

`_host_stage_body_transforms_if_needed` validates:
- `x_world_R_wp.shape == (num_envs, num_bodies, 3, 3)`
- `x_world_r_wp.shape == (num_envs, num_bodies, 3)`
- shapes agree on env/body dimensions
- `env_idx < num_envs`
- `max_body_index < num_bodies`

These checks guard against `GpuPublishedFrame` layout changes silently
producing wrong geometry. They are cheap (shape checks only, no data copy)
and run before any `.numpy()` call.

### 2.5 Test design: reading `body_z` from the frame

The body-bound tests read `body_z = float(frame.x_world_r_wp.numpy()[0, 0, 2])`
after `wp.synchronize_event(frame.ready_event)`. This makes the test
independent of the exact physics integration result — it checks that the
optical result is consistent with the published frame, not that the ball is
at a specific height.

This is the right pattern for GPU optical tests: assert consistency between
the optical result and the frame it was computed from, not absolute values
that depend on physics parameters.

## 3. Test coverage

| Test | What it checks |
|------|---------------|
| `test_execute_optical_on_gpu_published_frame_supports_body_bound_registry` | body-bound plane follows `x_world_r_wp`; range and position match body height |
| `test_body_bound_optical_result_survives_published_slot_reuse_after_completion` | body-bound device result is readable after 3 more engine steps |

## 4. Deferred

- Fully device-resident body transform reads (L5C).
- Device scene cache.
- GPU BVH.
- GPU direct-light and shadow rays.
- Multi-env batched optical runtime semantics.
