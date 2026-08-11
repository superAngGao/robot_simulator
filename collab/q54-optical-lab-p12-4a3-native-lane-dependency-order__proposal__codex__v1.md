# Q54 Optical Lab P12.4a3 Native-Lane Dependency Order Correction

Author: Codex

Date: 2026-08-11

Status: proposed correction for owner and Claude review

Parent design: P12.4 proposal v5 (`46a5d33`)

Parent implementation: P12.4a2 (`1824fd90c1c84fa5633800cd5ff12a8e9b5014c5`)

## Conclusion

Do not implement P12.4a3 by wrapping the current product executors and do not
run an a4 baseline from those wrappers. The approved a2 timed contract and the
current production paths are not compatible:

- a2 forbids allocation, compilation, host upload/materialization, target-wide
  synchronization, cross-lane waits, and Warp/Torch bridges inside R0-R2;
- the current Warp and CUDA render, acceleration, pack, and readback paths use
  those operations inside the proposed transaction.

Wrapping them would produce a benchmark that has the correct row names but the
wrong boundary. Hiding the same operations inside C++ or a caching allocator
would not make the measurement valid.

The recommended correction is to split a3 into an eligibility/audit slice and
a completed-lane slice, execute the already-approved native ownership/stream/
arena prerequisites between them, and keep a4 after the completed-lane slice.
This reorders existing work; it does not add a sixth optimization hypothesis or
expand retained product scope.

## Code evidence at HEAD

### Warp renderer is not preallocated

`optics/warp_execution.py` currently allocates inside every camera render:

- `GpuDeviceBvhOpticalExecutor.execute_camera(...)` allocates `origins` and
  `directions` with `wp.empty`;
- `_execute_device_ray_arrays(...)` allocates or uploads hit, range, position,
  normal, instance/material ID, and traversal diagnostic arrays;
- `GpuDeviceBvhDirectLightOpticalExecutor._shade_geometry(...)` allocates RGB,
  intensity, and shadow diagnostics.

The kernels themselves are valid production kernels, but the executor call is
not an R0 preallocated timed path.

### CUDA renderer allocates and globally synchronizes

`optics/cuda_direct_light.py` currently:

- allocates raygen output in `_generate_pinhole_camera_ray_tensors(...)`;
- calls extension functions whose C++ implementations allocate first-hit and
  shade outputs;
- calls `torch.cuda.synchronize(tensor_device)` after first hit and again after
  shade;
- uses `channel_to_torch(...)` for every scene/BVH input because the product
  executor still accepts Warp-owned scene/BVH objects.

Passing Torch tensors through the existing dataclasses avoids some Python-level
view conversion, but it does not remove output allocation or the two target-
wide synchronizations. It also is not yet the reviewed Torch-owned product
scene/BVH contract.

### Current acceleration paths invalidate R1/R2

`optics/device_bvh.py::build_device_bvh_from_snapshot(...)` synchronizes the
snapshot event, materializes AABBs and identity arrays on the host with
`.numpy()`, builds on the CPU, and uploads new Warp buffers. A topology built
before timing can be refit on the GPU, but that is not the current per-frame
build transaction described by the `cpu_bvh` identifier unless the gate
explicitly defines topology origin separately from timed update strategy.

`optics/cuda_lbvh.py::build_cuda_lbvh_from_snapshot(...)` currently:

- host-synchronizes the Warp snapshot event;
- calls `wp.to_torch` for AABBs;
- allocates Torch reduction, Morton-sort, topology, bounds, and metadata
  tensors inside the call;
- calls `torch.cuda.synchronize()`;
- calls `wp.from_torch` for every returned BVH view.

The frozen lane binding assigns `cuda_lbvh` to both production-static lanes and
to the CUDA dynamic lane. Therefore those R1/R2 rows cannot use the current
function inside timing.

### Current delivery paths invalidate R2

The retained R2 matrix includes synchronous and ordered asynchronous RGB/RGB8.
Today:

- `tools/optical_pipeline_lab/async_readback.py` resolves device channels with
  `channel_to_torch(...)` and owns Torch pinned-host allocation;
- `tools/optical_pipeline_lab/rgb_pack.py` allocates Warp RGB8 output;
- synchronous staging materializes host arrays and synchronizes readiness;
- CUDA RGB8/ordered-async product support is intentionally scheduled after
  C0/C2/C4 and B3.

Precreating output views and readback rings outside timing is valid. Creating
them lazily in the timed transaction is not.

### Canonical Go2 asset is absent from the clean test host

The gate requires `mujoco_menagerie/unitree_go2@run_manifest`, but
`out/external/mujoco_menagerie/unitree_go2` is not present on the host at this
HEAD. A synthetic or robot-like replacement must not be labeled as a Go2 row.
The exact asset and content manifest must be staged separately, using the
owner-approved local network relay if the server cannot fetch it directly.

## Rejected implementations

### Time the current product calls and record their wall time

Rejected. This violates the frozen native-ready boundary and mixes host waits,
allocation, and bridge creation into one lane differently from the other.

### Let internal C++ allocation escape the Python guard

Rejected. The a2 guard is an enforcement interface, not a definition of
fairness. An unobserved allocation remains an allocation.

### Copy Warp/CUDA kernels into `benchmarks/optical`

Rejected. The benchmark would compare benchmark-only implementations instead
of production kernels, and the selected durable lane could drift from product.

### Relax the a2 gate after seeing measurements

Rejected. The gate and receipt semantics were frozen before baseline precisely
to prevent this. Any relaxation requires a new owner decision before results,
and it still would not solve current bridge and target-wide synchronization.

### Use a surrogate scene for missing Go2 assets

Rejected. Resolution alone does not define the workload; primitive count,
material layout, BVH behavior, memory, and shadow work would all differ.

## Recommended corrected slice order

### P12.4a3a — real-lane eligibility and contamination audit

Implement no performance result writer yet. Add:

- actual Warp and CUDA lane adapters with row/case/renderer/acceleration
  identity, but require a preflight before `submit_sample(...)`;
- instrumentation at every concrete Python/runtime entry used by each lane,
  including allocation, upload/materialization, global synchronization,
  Warp/Torch/DLPack conversion, extension load, and cross-runtime wait;
- an immutable `LaneEligibilityReport` distinct from `ExecutionReceipt`;
- negative tests proving current invalid entry points are observed even when
  they occur below a higher-level executor call;
- acceleration attribution that separately records topology origin, timed
  update strategy, renderer, and any prebound cross-runtime acceleration view;
- exact canonical asset preflight and manifest failure for absent Go2.

An ineligible row must fail before timed submission and must not emit a result
record that `validate_result_record(...)` can accept. Eligibility artifacts are
diagnostic evidence only and cannot be treated as a baseline.

### P12.4c0a/c0b — execute the already-approved publication prerequisite

Proposal v5 states that C0 design approval and implementation/lifecycle tests
block C2-C4. Keep that rule. Complete the public CUDA consumer-stream binding
before pulling native CUDA ownership/stream/arena work forward.

This does not enable the dynamic CUDA product path by itself.

### P12.4c2/c3/c4 and B3 prerequisite portions

Implement the production-owned primitives required by valid native lanes:

- indexed Torch stream launch and Torch completion events without target-wide
  synchronization;
- Torch-owned CUDA scene update and LBVH outputs;
- reusable ray, geometry, shade, scene, acceleration, RGB8, and readback
  arenas with explicit leases;
- preallocated/out-style kernel entry points rather than allocation hidden in
  the extension;
- CUDA-native RGB8 and ordered delivery support;
- prebound Warp-output/Torch-readback views, where the retained Warp delivery
  contract legitimately uses them, created and validated outside timing.

These are the approved C2/C3/C4/B3 responsibilities, not new benchmark-only
kernels. They count against the existing optimization slices exactly once.
Profile-aware C1 and conditional fusion C5 remain in their approved positions;
C5 still requires valid R0/R1 evidence.

### P12.4a3b — complete valid Warp/CUDA R0-R2 lanes

After prerequisites pass:

- build independent Warp/Torch inputs from the same canonical logical case;
- preallocate all R0 output/diagnostic buffers;
- execute fresh scene-update/acceleration/render/lease work for R1;
- execute that same fresh R1 work plus payload-specific delivery/reclaim/drain
  for every supported R2 row;
- emit lane-owned readiness, input, acceleration, output, and lifecycle
  evidence;
- validate direct CPU/Warp/CUDA output parity outside primary samples;
- prove all instrumented contaminating operations remain absent in successful
  timed samples.

The Warp lane may invoke existing production Warp kernels through a thin
preallocated adapter; it must not copy them. The CUDA lane must use the same
production native primitives intended for the product path.

### P12.4a4 — comparator validity review, then common-subset baseline

Keep a4 after a3b. Its approval must check:

- no eligibility failure or uninstrumented runtime entry remains;
- the two lanes start from equivalent native-ready boundaries;
- R0 uses the frozen topology and primitive order;
- R1/R2 acceleration identity and update work match the gate;
- every R2 trace contains a fresh R1 transaction before delivery;
- asynchronous samples include final drain and reclaim;
- exact Go2 asset/case hashes match;
- profiled diagnostics are separate from unprofiled primary samples.

Only after that review may a baseline influence optimization or backend
selection.

## Gate and schema consequences

Do not change existing R0/R1/R2 row thresholds or case hashes.

Before a3a implementation, add versioned, pre-result contracts for:

- `lane_eligibility_schema_version`;
- concrete instrumented operation names and expected entry-point coverage;
- `topology_origin` versus `timed_acceleration_update`;
- `asset_manifest_sha256` for asset-backed cases;
- the rule that eligibility artifacts cannot satisfy a result row.

The gate remains frozen in the sense that no measured baseline has been
observed. This schema extension records newly discovered execution provenance;
it does not alter a performance threshold after results.

## Repository cleanup

Add a cleanup-ledger entry when a3a lane adapters land:

- retain the selected lane as the durable regression adapter;
- delete the losing lane after the decision;
- delete eligibility/paired-decision code that has no selected-backend
  regression value;
- retain concrete runtime instrumentation only where the selected lane's
  regression continues to enforce the same boundary;
- never add a product backend selector or production import of benchmarks.

## Acceptance requested

Please approve or request changes on these decisions before a3 code begins:

1. a3 is split into eligibility (`a3a`) and valid completed lanes (`a3b`);
2. no current executor wrapper may emit a baseline result while it violates the
   frozen timed contract;
3. C0a/C0b and the required C2/C3/C4/B3 native primitives move before a3b;
4. no benchmark-only kernel copy or surrogate Go2 scene is allowed;
5. a4 remains the first point where valid common-subset results may be
   inspected.
