# Q54 Optical Lab P12.4c3c CUDA Scene Update — Implementation Note (Codex v1)

Date: 2026-09-02
Base: `6d2a8d583a6966152631f50134241a1a3715f62a`
Scope: approved C3c only; no C3d/C4/B3/a3b activation or baseline results

## Owner summary

C3c now provides a Torch-owned CUDA scene-update data plane over the exact a2
prepared allocations. The implementation transforms canonical local triangles
and planes into caller-owned world geometry/AABB buffers on the frozen C2
transaction stream. It performs no timed tensor, event, workspace, extension,
or dynamic C++ Tensor-container allocation and records no terminal event; the
transaction owner remains the only caller of `finish()`.

This slice does not make a benchmark lane result-capable. C3d must still supply
the native CUDA LBVH rebuild, C4 must supply final arenas/slots/generations and
leases, B3 must supply native delivery, and C3e must close permanent runtime
instrumentation and comparison-lane integration before a3b can be activated.

## Implemented surface

### Torch-owned local and world scene storage

`optics/cuda_scene.py` now owns:

- `CudaOpticalLocalScene`, built as exact zero-copy wrappers over the a2
  Torch-owned scene allocations;
- `CudaSceneTransformInputs`, including the selected replay frame and its
  canonical replay provenance;
- `CudaWorldSceneBuffers`, whose triangle world geometry, plane world geometry,
  normals, and AABBs are allocated by the caller outside timing;
- `allocate_cuda_world_scene_buffers()` as the setup/test allocator that C4
  will replace with final arena slots.

The benchmark-native helpers bind the exact a2 manifest, packed-scene owner,
Torch allocations, replay digest, and replay frame index. They do not upload,
copy, bridge, or resolve readiness inside the timed submission.

### Fixed CUDA extension ABI

The versioned extension exposes exactly two out-style entries:

```text
update_triangles_world_aabb_out(...)
update_planes_world_out(...)
```

Both entries receive fixed positional Tensor parameters. The C++/CUDA source
contains no `std::vector<torch::Tensor>`, Torch allocating operation,
`cudaMalloc`, event creation, wait, or synchronization. It validates the
indexed device, dtype, rank/shape, contiguity, count range, and input/output
storage disjointness, installs `c10::cuda::CUDAGuard`, and launches on
`at::cuda::getCurrentCUDAStream(anchor.get_device())`.

Triangle update writes `v0`, both edges, normalized normal, and min/max AABBs.
Plane update writes normalized world normal and world point. A missing primitive
class binds no entry and launches no dummy kernel; a completely empty scene has
zero extension dispatches.

### Immutable transaction stage

`CudaSceneUpdateStageFactory.prepare()` loads the extension outside timing and
freezes the exact module, entry callables, target context/device/stream, native
manifest digest, input contract, local-scene identity, scene digest, and local
storage manifest. `bind()` freezes the selected transform view, replay
provenance/frame, environment index, caller-owned outputs, resources, exact
argument tuples, dispatch plan, and independent stage evidence.

Factory and stage public mutation is rejected. Construction-time snapshots are
rechecked so `object.__setattr__()` drift of factory fields, transform metadata,
entry identity, argument tuples, or storage is rejected before the first kernel
launch. A failure after transaction construction poisons the transaction and
cannot expose a completion ticket or call `finish()` successfully.

The seal is strictly one-way: after the construction baseline exists, ordinary
assignment cannot clear `_sealed`, replace the baseline, or mutate any other
field, and attribute deletion is always rejected. Repeated `_freeze()` calls
also fail. Transaction preflight reconstructs the formal triangle and plane ABI
and checks every Tensor argument by object identity plus the scalar environment
index by value. It independently reconstructs all stage evidence from the
factory, transforms, outputs, storage manifests, and dispatch plan, so
same-shape output permutation or stale evidence cannot be hidden by a replaced
signature baseline.

`CudaSceneUpdateStage.enqueue()` only enters the registered C2 stage, optionally
records already-provided diagnostic phase events, and invokes its prebound
non-empty entries. In the composed test, the ordered stage list is:

```text
cuda_scene_update -> cuda_direct_light_render -> owner finish
```

There is no scene-ready event, wait, host synchronization, or intermediate
terminal event.

## Verification

Environment evidence:

- `robot_sim` Conda environment;
- PyTorch `2.9.1+cu128`, CUDA runtime `12.8`;
- three visible GPUs: two A100-SXM4-40GB and one H200 NVL.

Focused CUDA scene tests cover:

- direct parity with the canonical dynamic replay host calculation;
- the asymmetric fixed fixture;
- plane-only, empty-triangle, and fully empty scenes;
- C3c plus C2 direct-light composition, with traversal diagnostics off/on;
- exact dtype/layout/storage-overlap rejection;
- factory, transform, callable/argument, and storage mutation rejection;
- transaction poison and absence of a completion ticket on failure;
- successful and failing extension calls on `cuda:1` while the caller ambient
  device is deliberately `cuda:0`, including restoration of the previous
  target-device current stream;
- source/ABI rejection of allocating and synchronization primitives;
- runtime timing hooks around the real stage and transaction finish.

Commands and results:

```text
python -m pytest -q tests/unit
1142 passed, 20 skipped in 55.10s

python -m pytest -q \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py
90 passed in 9.03s

python -m pytest -q tests/gpu/test_optical_cuda_scene.py
8 passed in 3.15s
```

Targeted `ruff check` and `git diff --check` pass.

## Gates intentionally unchanged

- `a3b_result_capable_before_C4_B3_review` remains `false`;
- `native_result_activation.activation_commit` remains `null`;
- no lane gains a result-capable C4 binding or `submit_sample()` path;
- no benchmark result, baseline, threshold, or backend-selection artifact is
  produced;
- Warp optical remains intact for the paired comparison and terminal decision.

## Next slice

C3d should consume `CudaWorldSceneBuffers.triangle_aabb_min/max` and add the
approved caller-workspace, fixed-ABI Morton/Karras rebuild transaction stage.
It must preserve this same context/stream and defer the sole terminal event to
the complete transaction owner's `finish()`.
