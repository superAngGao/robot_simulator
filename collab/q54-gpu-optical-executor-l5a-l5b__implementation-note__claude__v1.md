Initiative: q54-gpu-optical-executor
Stage: l5a-l5b0-implementation-note
Author: claude
Version: v1
Date: 2026-05-03
Status: implemented-l5b0
Related Files: optics/device.py, optics/warp_execution.py, optics/gpu_runtime.py, optics/execution.py, optics/__init__.py, tests/unit/optics/test_device_optical.py, tests/gpu/test_optical_warp_executor.py, tests/gpu/test_optical_gpu_runtime.py

# Q54 L5A + L5B.0 GPU Optical Executor — Claude Implementation Note

## 1. Scope

L5A: Warp brute-force first-hit executor (`optics/warp_execution.py`,
`optics/device.py`).

L5B.0: Q52 `GpuPublishedFrame` borrow/complete lifecycle helper
(`optics/gpu_runtime.py`), plus `OpticalComputeResult.resources` field
(`optics/execution.py`) to prevent premature GC of kernel input buffers.

## 2. 关键思考

### 2.1 Host-side role filtering vs. device-side bitmask

**Decision**: filter by `sensor_role` on the host before uploading to the
device. The kernel receives only visible primitives.

**Alternatives considered**:
- Pass a per-primitive role bitmask to the kernel and branch inside the loop.
  Rejected: adds a branch per primitive per ray, and role strings cannot be
  passed to Warp kernels without extra encoding.
- Build a separate registry per role. Rejected: duplicates geometry data and
  complicates the registry API.

Host filtering is the right split: it keeps the kernel focused on first-hit
math, and role filtering is a one-time O(instances) cost per call, not per ray.

### 2.2 Packed int64 source-order key for tie-breaking

**Decision**: `key = instance_index * 2**32 + primitive_index_within_instance`
as a single `int64` comparison in the kernel.

**Why**: Warp kernels cannot compare Python strings or tuples. A single integer
comparison is the minimal encoding that preserves CPU lexicographic source order
with no extra memory and no branching overhead.

**Risk**: `instance_index * 2**32` overflows `int64` when `instance_index >=
2**31`. `pack_source_order_key` guards this with an explicit range check. The
constant `MAX_PRIMITIVES_PER_INSTANCE = 2**32` is documented as the per-instance
primitive budget.

### 2.3 `OpticalComputeResult.resources` — preventing async GC

**Problem discovered during L5B.0 testing**: the published-slot reuse test
(`test_optical_result_buffers_survive_published_slot_reuse_after_completion`)
failed intermittently. After `complete_device_consumer(...)`, the engine was
free to reuse the published slot. The Python GC could then collect the Warp
input arrays (origins, directions, triangles, …) before the optical stream had
finished consuming them, causing silent data corruption.

**Fix**: `OpticalComputeResult` gains a `resources: tuple[object, ...]` field
(frozen dataclass, `repr=False`, `compare=False`). The executor stores all
kernel input arrays there. The result object now owns the lifetime of every
device buffer the kernel touches.

**Alternative considered**: require callers to synchronize before the result
goes out of scope. Rejected: this is an invisible contract that is easy to
violate and hard to test.

**Alternative considered**: synchronize inside `execute(...)` before returning.
Rejected: defeats the purpose of async GPU execution.

The `resources` field is the minimal, explicit lifetime anchor. It costs one
tuple allocation per call.

### 2.4 L5B.0 world-static restriction

**Decision**: raise `NotImplementedError` if any registry instance has
`body_index is not None`.

**Rationale**: body-bound GPU optical packing requires reading
`GpuPublishedFrame.x_world_*` transforms directly on the device. L5B.0 fakes
body transforms by constructing an empty `CpuPublishedFrame` with no bodies.
Silently accepting body-bound instances would produce wrong geometry without any
error. An explicit guard is safer than a silent wrong answer.

**Next step**: when the device scene packer can read `x_world_*` directly, the
guard becomes a capability check rather than a hard rejection.

### 2.5 `_synchronize_ready_event` in staging

`stage_optical_compute_result_to_host` calls `wp.synchronize_event(ready_event)`
before copying device channels to NumPy. This makes staging safe when the caller
does not globally synchronize the device first.

The `try/except` import guard inside `_synchronize_ready_event` is intentional:
`stage_optical_compute_result_to_host` is in `optics/device.py`, which must
remain importable in CPU-only environments. Warp is an optional dependency.

## 3. Test coverage

| Test file | Coverage |
|-----------|---------|
| `tests/unit/optics/test_device_optical.py` | pack_source_order_key, role filter, workload shape/dtype, staging normalization |
| `tests/gpu/test_optical_warp_executor.py` | plane/miss, triangle, tie-break, role filter, camera postprocess — all vs. CPU reference |
| `tests/gpu/test_optical_gpu_runtime.py` | Q52 consumer lifecycle, ready_event, buffer survival after slot reuse, body-bound rejection, timeline mismatch rejection |

## 4. Deferred

- Body-bound GPU optical scene packing (read `x_world_*` on device).
- Device scene cache (avoid re-packing static geometry every call).
- Result buffer pooling.
- GPU BVH.
- GPU direct-light / shadow rays.
- Multi-env batched optical runtime.
