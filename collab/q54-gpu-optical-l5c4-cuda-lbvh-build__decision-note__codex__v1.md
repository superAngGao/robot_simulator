Initiative: q54-gpu-optical-l5c4-cuda-lbvh-build
Stage: decision-note
Author: codex
Version: v1
Date: 2026-05-06
Status: draft
Related Files: optics/device_bvh.py, examples/mujoco_menagerie_gpu_preview.py, OPEN_QUESTIONS.md, setup.py
Owner Summary: After timing the Go2 GPU preview path, the slow step is not GPU traversal or host/device copies. The initial BVH build spends about 11.7s in the pure-Python CPU median-split builder. The next performance step should add an optional CUDA LBVH build backend while keeping the current Warp traversal/direct-light path as the correctness reference.

# Q54 L5C.4 CUDA LBVH Build Decision

## Current Finding

Go2 Menagerie preview timing, 398,432 triangles:

```text
device_to_host ~= 2.6 ms
upload         ~= 6.2 ms
host_build     ~= 11.7 s
```

The rebuild bottleneck is therefore the Python BVH builder, not PCIe transfer
or the current Warp traversal/shading kernels.

An experimental `argpartition` median split only reduced host build from about
11.7s to about 10.9s. That suggests the dominant cost is the full Python build
shape: recursive node construction, repeated per-node bounds reductions,
array slicing, Python objects, and breadth-first remapping.

## Decision

Do not rewrite the full GPU optical renderer yet.

Add a CUDA BVH build backend first, and keep the current Warp BVH traversal and
direct-light/shadow executor as the correctness path.

The intended backend split is:

```text
Warp:
  - device scene update/refit correctness path
  - first-hit traversal
  - direct-light and shadow parity path
  - existing GPU tests and semantic contract

CUDA:
  - Morton code generation
  - radix sort, preferably CUB
  - LBVH topology generation
  - bottom-up or level-by-level bounds build
  - output arrays compatible with DeviceOpticalBvh
```

This keeps the first CUDA step focused on the measured bottleneck.

## Initial CUDA Builder Contract

Input:

```text
triangle_aabb_min: float32[num_triangles, 3]
triangle_aabb_max: float32[num_triangles, 3]
source_order_key:  int64[num_triangles]
primitive metadata arrays already present in DeviceOpticalScene
```

Output should match the existing `DeviceOpticalBvh` layout:

```text
bounds_min: float32[num_nodes, 3]
bounds_max: float32[num_nodes, 3]
left:       int32[num_nodes]
right:      int32[num_nodes]
start:      int32[num_nodes]
count:      int32[num_nodes]
node_depth: int32[num_nodes]
level_ranges: int32[max_depth + 1, 2]
prim_ids:   int32[num_triangles]
primitive metadata arrays
```

The first CUDA backend may produce a different tree topology from the CPU
median-split builder. That is acceptable if traversal results preserve the
observable contract:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
material_index / direct-light RGB where applicable
role filtering
source-order tie-break for equal-distance hits
stack overflow diagnostics
Q52 resource lifetime
```

## Algorithm Direction

Recommended first implementation: GPU LBVH.

1. Compute scene bounds from triangle centroids or AABBs.
2. Compute Morton code per primitive.
3. Sort `(morton_code, primitive_id)` by Morton code using CUB radix sort.
4. Build LBVH internal topology in parallel.
5. Build leaf bounds from primitive AABBs.
6. Build internal bounds bottom-up or level-by-level.
7. Fill `level_ranges` and diagnostics.

Do not start with GPU SAH. SAH/spatial split remains a future quality/perf
comparison after LBVH is working.

### Detailed LBVH Shape

The intended first CUDA builder is a Karras-style binary radix tree over sorted
Morton keys:

```text
triangle AABB
  -> centroid
  -> scene-normalized centroid
  -> 30-bit Morton code, 10 bits per axis
  -> deterministic extended key
  -> CUB radix sort
  -> parallel internal-node range/split generation
  -> leaf bounds from primitive AABBs
  -> internal bounds reduction
  -> DeviceOpticalBvh-compatible SoA arrays
```

The deterministic key should be:

```text
extended_key = (morton_code << 32) | primitive_id
```

The low bits are not a source-order key; they only make duplicate Morton codes
deterministic. `prim_ids` must still store original global primitive ids, and
source-order tie-break must keep using:

```text
primitive_source_order_key[prim_id]
```

The first topology implementation may use leaf size 1. Leaf grouping can be
added after parity and initial benchmarks.

`node_depth` / `level_ranges` are the main integration decision. Existing Warp
traversal only needs `left/right/start/count/prim_ids`, but existing GPU refit
uses `level_ranges`. The build plan is therefore:

1. First spike: Morton keys + sorted primitive ids.
2. Second spike: binary radix tree topology + traversal parity; refit may be
   marked unsupported if level metadata is not valid yet.
3. Third step: fill true `node_depth` and contiguous `level_ranges`, or reorder
   nodes level-by-level, so CUDA-built BVHs support the same refit path as
   CPU-built BVHs.

### External Alignment

The design aligns with common public BVH construction patterns:

- NVIDIA/Karras LBVH: maximize GPU parallelism by building internal nodes from
  sorted Morton keys rather than doing top-down recursive splits.
  Reference: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
- ToruNiina/lbvh: CUDA/Thrust LBVH implementation based on Karras 2012; uses
  extended indices for duplicate Morton codes. This matches the
  `morton_code + primitive_id` deterministic ordering requirement.
  Reference: https://github.com/ToruNiina/lbvh
- pbrt HLBVH: uses Morton-built treelets plus SAH upper construction. This is a
  future option if pure LBVH build speed is good but traversal quality is not.
  Reference: https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
- Warp.Bvh: documents LBVH as the default GPU construction method and SAH as the
  default CPU-oriented construction method, matching our LBVH-first CUDA
  direction while keeping SAH/Embree/TinyBVH as quality baselines.
  Reference: https://nvidia.github.io/warp/api_reference/_generated/warp.Bvh.html

## Integration Options To Spike

Current `setup.py` is pure Python and there is no existing `.cu/.cpp`
extension directory. The first spike should compare:

1. `torch.utils.cpp_extension` JIT extension:
   - fastest path to compile CUDA/CUB in the current environment;
   - can return Torch CUDA tensors;
   - needs interop or copying into Warp arrays.

2. Small package extension in `setup.py`:
   - cleaner install story;
   - more packaging work;
   - still optional under an extra such as `cuda-bvh`.

3. External benchmark binary:
   - useful for CUB/LBVH algorithm validation;
   - not enough for runtime integration by itself.

The first spike should prefer option 1 unless it conflicts with the repo's
packaging constraints.

## Review Follow-Up Decisions

### Torch/Warp Interop

The first spike should use `wp.from_torch` and shared CUDA device pointers, not
D2H/H2D copies. Ownership must be explicit:

```text
Torch tensors own CUDA allocations.
Warp arrays are non-owning views.
DeviceOpticalBvh.resources must retain the owning Torch tensors or a wrapper
for as long as the Warp traversal can read the node arrays.
```

Stream synchronization is part of the spike, not a later integration detail.
The first implementation may conservatively synchronize between the CUDA
extension and Warp traversal while proving correctness, then replace that with
stream-aware handoff once the backend boundary is stable.

### `level_ranges` / `node_depth`

Current Warp traversal consumes `left/right/start/count/prim_ids` and does not
use `level_ranges`. Existing GPU refit does use `stats.level_ranges` for
level-by-level bounds propagation.

Therefore CUDA LBVH build must either:

1. fill true `node_depth` and contiguous `level_ranges`, and support existing
   refit; or
2. explicitly mark refit unsupported for that BVH instance until the metadata
   is filled.

The preferred first integrated backend is option 1 so that CUDA-build BVHs keep
the same operational contract as CPU-build BVHs.

### Source-Order Tie-Break

Morton sort may reorder leaves, but `prim_ids` must always contain the original
global primitive id. Source-order tie-breaks must continue to read:

```text
primitive_source_order_key[prim_id]
```

The sorted index is not a source-order key.

### Test Isolation

CUDA extension tests must not run in ordinary CPU or Warp-only CI. Use
`pytest.importorskip`, `@pytest.mark.cuda_ext`, and explicit availability checks
for CUDA, nvcc, Torch, and CUB. Extension compile failure should skip the spike
test unless the user explicitly requested CUDA extension validation.

### Public API Boundary

Do not change the external `DeviceOpticalBvh` shape or force callers to know
which backend produced the arrays. CUDA build should be an internal backend
choice, for example through a backend argument or dispatch helper. Existing
callers should keep receiving a `DeviceOpticalBvh` with the same public fields.

## Spike Result: CUDA/CUB + Torch/Warp Interop

Added:

```text
examples/cuda_lbvh_extension_spike.py
```

The spike JIT-compiles a minimal CUDA extension with
`torch.utils.cpp_extension.load_inline`, includes CUB, calls
`cub::DeviceRadixSort::SortPairs`, returns Torch CUDA tensors, and then exposes
the sorted primitive id tensor to Warp using `wp.from_torch`.

Observed on 2026-05-06:

```text
nvcc:  /usr/local/cuda/bin/nvcc
torch: 2.9.1+cu128, CUDA available, torch CUDA 12.8
warp:  1.12.0
GPU:   NVIDIA H200
```

Result:

```text
sorted_codes: [0, 3, 3, 9, 9, 17, 31, 42]
sorted_primitive_ids: [6, 1, 3, 4, 7, 0, 5, 2]
warp_from_torch_plus_one: [7, 2, 4, 5, 8, 1, 6, 3]
```

This validates the first integration path:

```text
CUDA/CUB extension -> owning Torch CUDA tensors -> non-owning Warp views
```

The first JIT compile is expensive and should remain outside default tests.
After cache warm-up, the standalone spike process still took about 14.6s
end-to-end due to Python/Torch/Warp extension startup, so production preview
should not rebuild or re-JIT in ordinary image generation paths.

The spike was then extended from hand-written code sorting to AABB-derived
Morton sorting:

```text
triangle AABB -> centroid -> 30-bit Morton code -> extended_key -> CUB sort
```

Observed result:

```text
morton_sorted_keys:
  [1002720244793344, 659528516479483907, 1318054312714174466,
   2635105905183555585, 4035225266123964420]
morton_sorted_primitive_ids:
  [0, 3, 2, 1, 4]
```

The spike verifies that:

```text
sorted_keys is monotonic
sorted_primitive_ids is a permutation of original primitive ids
low_32_bits(sorted_key) == primitive_id
```

This covers the first LBVH invariant needed before topology construction:
Morton ordering is deterministic and primitive identity is not lost.

The spike was then extended with minimal Karras-style topology construction:

```text
sorted extended keys
  -> one CUDA thread per internal node
  -> range direction + range length from common-prefix tests
  -> split from highest differing bit
  -> full node arrays with internal nodes [0, n-2] and leaves [n-1, 2n-2]
```

Observed result for five primitives:

```text
topology_left:   [2, 4, 1, 7, -1, -1, -1, -1, -1]
topology_right:  [3, 5, 6, 8, -1, -1, -1, -1, -1]
topology_parent: [-1, 2, 0, 0, 1, 1, 2, 3, 3]
```

The Python-side spike validator checks:

```text
root parent == -1
each internal child points back to the parent
each leaf has start=leaf_rank and count=1
all nodes are reachable exactly once from root
```

This validates the second LBVH invariant: sorted keys can produce a valid
binary radix topology. The next step is bounds construction and conversion into
the exact `DeviceOpticalBvh` layout consumed by Warp traversal.

The spike was then extended with leaf and internal bounds construction:

```text
leaf bounds:
  sorted_prim_ids[leaf_rank] -> primitive AABB -> leaf node bounds

internal bounds:
  parent links + atomic child counters
  each child atomically contributes min/max to the parent
  second arriving child continues upward
```

Observed root bounds for the sample AABBs:

```text
root_bounds_min: [0.0, 0.0, 0.0]
root_bounds_max: [0.550000011920929, 0.550000011920929, 0.550000011920929]
```

The spike validator checks root bounds against the min/max of all primitive
AABBs and each leaf bound against the AABB of its sorted original primitive id.
This validates the third LBVH invariant: topology and primitive ordering are
sufficient to build usable node AABBs on the GPU.

The next implementation step is to convert these tensors into a
`DeviceOpticalBvh`-compatible object:

```text
Torch owning tensors -> wp.from_torch non-owning views -> DeviceOpticalBvh.resources retains owners
```

## First Integrated Builder

Added:

```text
optics/cuda_lbvh.py
build_cuda_lbvh_from_snapshot(snapshot, ...)
```

The function builds the CUDA LBVH tensors, wraps them with `wp.from_torch`, and
returns a real `DeviceOpticalBvh` consumed by the existing Warp traversal
executor.

Current contract:

```text
supported:
  - CUDA LBVH build
  - leaf-size-1 traversal
  - Torch-owned tensor lifetime retained through DeviceOpticalBvh.resources
  - Warp first-hit traversal parity

not yet supported:
  - GPU refit on CUDA-built topology
  - true node_depth / level_ranges
  - direct integration into preview default path
```

The returned stats use:

```text
split_strategy = "cuda_lbvh"
supports_refit = False
```

`refit_device_bvh_from_snapshot(...)` now rejects BVHs with
`supports_refit=False`, so callers do not accidentally treat the first CUDA
LBVH topology as level-refittable.

Parity added:

```text
tests/gpu/test_optical_gpu_runtime.py::test_cuda_lbvh_executor_matches_cpu_bvh_for_world_static_triangle_mesh
```

This compares:

```text
CPU median BVH + Warp traversal
CUDA LBVH + Warp traversal
```

for hit mask, range, position, normal, numeric instance id, and stack
diagnostics.

Implementation note: the first integrated builder still reuses the spike
extension source. Before promoting the backend beyond experimental status, move
the CUDA source strings out of `examples/` into an internal extension module or
package extension boundary.

## Go2 Preview Timing

`examples/mujoco_menagerie_gpu_preview.py` now supports:

```text
--bvh-backend cpu|cuda_lbvh
```

Go2 Menagerie, 398,432 triangles, shadows on:

```text
320x220:
  bvh_build ~= 512 ms
  CUDA build window ~= 3.1 ms
  warmed render p50 ~= 2.1 ms

960x640:
  bvh_build ~= 473 ms
  CUDA build window ~= 3.35 ms
  warmed render p50 ~= 22.6 ms
  total ~= 12.19 s
```

Prior CPU median-build preview timing was:

```text
960x640:
  bvh_build ~= 11.37 s
  total ~= 23.44 s
```

So the tree build path has moved from an 11s Python bottleneck to about 0.5s
end-to-end in the current experimental CUDA integration, with the core build
window in the low milliseconds. The remaining end-to-end preview bottleneck is
now dominated by `device_scene_snapshot` cold behavior, not BVH tree
construction.

## Parity And Benchmark Gates

Correctness:

- CPU median BVH + Warp traversal vs CUDA LBVH + Warp traversal.
- Cover first-hit, role mask, source-order tie-break, planes plus triangles,
  direct-light no-shadow, direct-light with shadow any-hit.

Performance:

- report CUDA build time separately from traversal and staging;
- compare CPU Python median build, optional native/TinyBVH baseline, CUDA LBVH,
  and later OptiX GAS/TLAS;
- include Go2 Menagerie and synthetic large meshes.

## Non-Goals For This Step

- CUDA traversal rewrite.
- CUDA direct-light/shadow rewrite.
- path tracing.
- OptiX adapter.
- shared collision/optical live BVH.
- changing the public `DeviceOpticalBvh` consumer API.
