# Q54 Optical Lab P12.4c3d Native CUDA LBVH — Implementation Note (Codex v1)

Date: 2026-09-03
Base: `6a5213e025018204dc30bb02e8bb41809b7a9374`
Scope: approved C3d only; no C3e/C4/B3/a3b activation or baseline results

## Owner summary

C3d now provides the caller-owned native CUDA LBVH rebuild required by the
CUDA candidate transaction. It consumes the world triangle AABBs written by
C3c, computes scene bounds, deterministic Morton/source-index ordering,
Karras topology, leaf bounds, and internal bounds on the same indexed Torch
CUDA stream, and writes every result into caller-owned tensors. The stage
records no ready or terminal event; the complete C2 transaction owner remains
the only caller of `finish()`.

Production extension source no longer lives in `examples/`. The old
Warp-snapshot builder remains only as an explicitly allocating, bridging, and
blocking compatibility facade so existing product behavior remains covered
until the terminal backend cleanup. The example is now a thin caller of the
production API.

This slice does not make either benchmark lane result-capable. C3e must still
deliver the static Warp comparison binding and permanent audit closure, C4
must provide final arena slots/generations/leases and C4-only bindings, and B3
must provide native delivery before a3b can be activated.

## Implemented surface

### Production CUDA/CUB source

`optics/_cuda_lbvh_source.py` owns the versioned extension source. It exposes:

```text
cuda_lbvh_workspace_requirements(primitive_count, device_ordinal)
cuda_lbvh_rebuild_out(20 fixed positional Tensor arguments)
```

The out-style entry performs this ordered operation on the indexed current
stream:

```text
world AABBs
-> six CUB scene-bound reductions
-> morton30/source-index composite keys
-> CUB radix-sort pairs
-> leaf metadata
-> Karras internal topology
-> typed node-bound initialization
-> leaf bounds
-> internal bounds
```

The composite key is exactly `(morton30 << 32) | source_index`, so equal
Morton codes retain a deterministic canonical primitive order. Device guards
and `at::cuda::getCurrentCUDAStream(anchor_device)` are used by both the query
and rebuild entry.

The timed source contains no Torch allocating operation, dynamic Tensor
container, `cudaMalloc`/`cudaFree`, event creation, host query/wait, or CUDA
device/stream synchronization. CUB receives the caller-owned workspace.
Integer sentinels/counters use same-stream `cudaMemsetAsync`; floating-point
infinities use a fixed initialization kernel.

Zero primitives bypass the extension and CUB entirely. A single primitive
produces one root/leaf and skips the internal-topology and internal-bounds
kernels.

### Workspace and output ownership

`optics/cuda_lbvh.py` now owns immutable workspace requirements including:

- extension ABI and source digest;
- device ordinal and compute capability;
- primitive and node capacity;
- CUB reduction/radix requirements;
- required byte count and alignment.

`CudaLbvhWorkspace` binds the exact aligned storage and its requirements.
`CudaLbvhBuffers` contains every mutable scene-bound, key/id, topology, range,
node-bound, and child-counter output. The setup helpers allocate one-shot
storage outside timing; C4 remains responsible for replacing these helpers
with final arena storage.

Bind rejects wrong device, dtype, shape, layout, primitive/node capacity,
workspace requirements, alignment, input/output overlap, and output/output
overlap before any enqueue.

### Immutable transaction stage and provenance

`CudaLbvhRebuildStageFactory.prepare()` loads the extension and performs the
CUB workspace query outside timing. It freezes the exact C2 execution context,
module/entry identity, device/stream, primitive count, requirement digest,
a2 native-input manifest digest, scene digest, and input contract.

`benchmarks.optical.build_prepared_cuda_lbvh_rebuild_factory()` mechanically
binds those values to the exact result-capable a2 prepared input and its
`PackedOpticalSceneHost` owner. Production optics code does not import the
benchmark package.

`bind()` freezes the exact C3c AABB tensors, all 17 output tensors, workspace,
fixed 20-position ABI tuple, resource owners, and independent
`CudaLbvhRebuildStageEvidence`. Evidence includes the formal acceleration,
operation, topology-origin/lifetime, storage and workspace digests, counts,
and dispatch-plan digest, but contains no C4 slot, generation, ring, lease, or
result-capable claim.

Factory and stage seals are one-way. Ordinary field mutation/deletion is
rejected, and construction snapshots, actual ABI object identities, storage
manifests, resources, and reconstructed evidence are checked again at
transaction preflight. Post-construction drift therefore fails before the
first enqueue.

`CudaLbvhRebuildStage.enqueue()` only enters its registered
`cuda_lbvh_rebuild` stage, optionally records preallocated diagnostic events,
and invokes its already-bound extension entry. It does not size/allocate a
workspace, create a container/event, synchronize, or return a completion
ticket.

The composed native test executes exactly:

```text
cuda_scene_update
-> cuda_lbvh_rebuild
-> cuda_direct_light_render
-> owner finish (one terminal event)
```

### Compatibility and cleanup

`build_cuda_lbvh_from_snapshot()` now delegates computation to the native
stage but deliberately retains all compatibility costs outside the native
enqueue: Warp readiness wait, Warp-to-Torch views, one-shot allocation,
dedicated Torch stream/event ownership, terminal host wait, Torch-to-Warp
views, and legacy `DeviceOpticalBvh` wrapping. It remains benchmark-ineligible.

The compatibility facade uses a non-default owned Torch stream because C2
correctly rejects the implicit default stream's zero handle. Its host wait
completes the native transaction before publishing the legacy Warp ready
event, preserving current product correctness without pretending to be the
native benchmark path.

`examples/cuda_lbvh_extension_spike.py` contains no C++/CUDA source,
`load_inline`, or alternate builder. It is a runnable thin smoke caller.

The architecture inventory and mandatory cleanup ledger now name the
production source. At P12.4e the CUDA-selected branch must remove the legacy
facade/Warp wrapping/experiment filename; the Warp-selected branch must either
pass the separately frozen acceleration-retention gate and productionize the
native API or delete the complete module/source/example surface.

## Verification

Environment:

- `robot_sim` Conda environment;
- PyTorch `2.9.1+cu128`, CUDA runtime `12.8`;
- visible H200 NVL and A100-SXM4 devices.

Focused C3d tests cover:

- fixed source ABI and absence of allocating/synchronizing primitives;
- thin-example ownership;
- asymmetric boxes, equal Morton codes, and fully degenerate scene extents;
- deterministic key/id order, full topology, parent/child relationships, and
  every leaf/internal descendant bound;
- even, odd, and multi-block primitive counts (`2`, `4`, `17`, `257`);
- explicit zero/single-primitive dispatch behavior;
- stale output overwrite and repeated deterministic rebuilds;
- wrong dtype/shape/layout, overlap, misaligned workspace, immutable-factory,
  immutable-stage, provenance, and ABI-position rejection;
- real runtime timing hooks with no observed timed contamination;
- success and direct-extension failure on `cuda:1` while ambient device is
  deliberately `cuda:0`, including ambient device and target current-stream
  restoration;
- C3c -> C3d -> C2 composition with one terminal event;
- existing CPU-BVH/Warp-BVH/CUDA first-hit and camera/direct-light parity via
  the compatibility facade.

Commands and results:

```text
python -m pytest -q tests/unit
1146 passed, 20 skipped in 54.39s

python -m pytest -q \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_cuda_lbvh_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py
103 passed in 8.21s

python -m pytest -q \
  tests/unit/optics/test_cuda_lbvh_native.py \
  tests/gpu/test_optical_cuda_lbvh_native.py
17 passed in 68.38s (cold extension rebuild)

python examples/cuda_lbvh_extension_spike.py --primitive-count 8
completed with one cuda_lbvh_rebuild stage and 15 output nodes
```

Targeted `ruff check`, repository inventory validation, and
`git diff --check` pass.

## Gates intentionally unchanged

- `a3b_result_capable_before_C4_B3_review` remains `false`;
- `native_result_activation.activation_commit` remains `null`;
- no C4 binding type, slot, generation, lease, arena, or `submit_sample()` path
  is introduced;
- no benchmark result, baseline, threshold, or backend-selection artifact is
  produced;
- Warp optical remains intact for the paired comparison and terminal decision.

## Next slice

C3e should implement the approved static Warp-owned/Torch-view comparison
binding on one native stream, prove zero timed bridge creation, update the
permanent a3a hook/manifest coverage, close the remaining C3 cleanup entries,
and keep a3b closed for C4/B3 review.
