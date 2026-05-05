Initiative: q54-gpu-optical-l5c2-gpu-bvh
Stage: review-followup
Author: codex
Version: v1
Date: 2026-05-05
Status: review-incorporated
Related Files: collab/q54-gpu-optical-l5c2-gpu-bvh-plan__review-request__codex__v1.md, OPEN_QUESTIONS.md
Owner Summary: Follow-up after Claude review of the L5C.2 GPU BVH plan. We accept the staged CPU-build/GPU-traverse -> GPU refit/LBVH -> OptiX route, and incorporate stronger requirements around primitive metadata, stack/register pressure instrumentation, plane pass order, TinyBVH external alignment, and primary-first-hit-only scope for L5C.2a.

# Q54 L5C.2 GPU BVH Plan Review Follow-Up

## Accepted Review Changes

Claude accepted the overall staged L5C.2 route:

```text
L5C.2a: CPU-built / GPU-traversed BVH
L5C.2b: GPU refit / GPU LBVH build
L5C.2c: OptiX adapter spike
```

We accept the requested plan refinements below.

## 1. Primitive Metadata Is An L5C.2a.0 Requirement

The global BVH may only need global primitive ids for first traversal, but
L5C.2a.0 must already materialize:

```text
primitive_global_id
primitive_instance_index
primitive_index_within_instance
primitive_geometry_index_or_id
primitive_geometry_primitive_index
primitive_source_order_key
```

This avoids a payload-breaking change when moving toward BLAS/TLAS.

## 2. Split Build And Traversal Metrics

L5C.2a benchmark output must split:

```text
cpu_build_ms
gpu_traverse_ms
staging_ms
```

Merged execute time is not enough to judge whether GPU traversal is improving.

## 3. Stack And Register Pressure

Initial stack policy changes from:

```text
MAX_BVH_STACK = 64
```

to:

```text
MAX_BVH_STACK = 32
```

L5C.2a must record:

```text
CPU build max tree depth
observed traversal max stack depth
overflow count
```

First implementation should use a node-id-only stack. Shared-memory/per-warp
stacks, stackless traversal, or wide BVH should be considered only after
measurements show the 32-entry stack is insufficient.

## 4. Plane Pass Order

Planes remain outside the BVH. The recommended first kernel order is:

```text
for each ray:
  best = analytical plane pass
  mesh_best = BVH triangle traversal
  best = merge(best, mesh_best, _is_better_hit)
  write result
```

This keeps plane/triangle tie-break parity easier to reason about.

## 5. GPU Refit Metadata

The first GPU refit design remains level-by-level rather than atomic
bottom-up. Build metadata must include:

```text
node_parent
node_level/order
level_ranges: int32[num_levels, 2]
leaf_node_ids
```

## 6. LBVH Sort Dependency

Before starting L5C.2b, explicitly evaluate:

```text
Warp sort utilities
Torch/CUDA sort as a temporary backend
design-only blocker if no reliable sort path exists
```

Do not leave the sorting dependency as an implementation-time surprise.

## 7. External Alignment Order

First external target:

```text
TinyBVH
```

Use Embree for CPU BVH build-quality sanity when practical. Treat pbrt-v4
GPU/OptiX as aspirational until scene conversion and phase-matched ray-query
benchmarking are practical.

## 8. BVH Quality Metric

Add a backend-independent SAH-style quality diagnostic:

```text
sum over leaves:
  (leaf_surface_area / root_surface_area) * leaf_primitive_count
```

This helps compare median split, LBVH, SAH, and external BVH quality before
ray distribution effects dominate.

## 9. Primary First-Hit Scope

L5C.2a benchmarks and parity should stay primary first-hit only.

Shadow/visibility rays have any-hit and early-termination semantics. They
should be added after the first-hit correctness bridge is stable.

## Updated Plan

The accepted changes have been incorporated into:

```text
collab/q54-gpu-optical-l5c2-gpu-bvh-plan__review-request__codex__v1.md
OPEN_QUESTIONS.md
```
