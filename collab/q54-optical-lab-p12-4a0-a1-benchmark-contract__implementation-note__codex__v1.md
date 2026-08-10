Initiative: q54-optical-lab-p12-4a0-a1-benchmark-contract
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-08-10
Status: implemented; awaiting Claude review
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v5.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v2.md; optics/output_contract.py; optics/device.py; optics/execution.py; optics/cuda_direct_light.py; optics/warp_execution.py; tools/optical_pipeline_lab/delivery.py; tools/optical_pipeline_lab/timing.py; benchmarks/optical/cases.py; benchmarks/optical/schema.py; benchmarks/results/q54_p12_4/gate.json; benchmarks/results/q54_p12_4/architecture_inventory.json; benchmarks/results/q54_p12_4/cleanup_ledger.json
Owner Summary: P12.4a0/a1 records the verified pre-selection architecture and two-terminal cleanup obligations, then freezes backend-independent workloads, timing boundaries, formal payload channels, row-bound execution/lifecycle evidence, comparison controls, and supported decision gates before any new Warp/CUDA baseline is observed. The reviewed correction removes an unfounded numeric memory limit and keeps only exact clean single-lane measurement plus mandatory-workload OOM failure. It adds no runner, backend adapter, product selector, CUDA publication API, or benchmark result.

## Scope and Ordering

This implementation follows approved proposal v5 and stops at a0/a1:

```text
P12.4a0 architecture inventory + cleanup ledger
P12.4a1 canonical cases + versioned schemas + frozen gate
```

P12.4a2 and later work is intentionally absent. In particular, C2-C4 and the
dynamic CUDA product path remain blocked until C0a is separately approved and
C0b has implemented and passed the required GPU lifecycle tests.

## Architecture Inventory

`architecture_inventory.json` records the code state observed at clean commit
`46a5d33` rather than assuming the physics backend from review prose:

- physics is a Warp-on-CUDA `GpuEngine` producer with CUDA-resident Warp arrays;
- `PublishedRing` owns logical consumer/reclaim state while `GpuEngine` owns
  the current Warp stream/event operations;
- the current complete optical product/oracle lane is Warp;
- the CUDA optical executor owns Torch CUDA raygen/traversal/direct-light but
  still consumes Warp-prepared scene/BVH state and returns a blocking result;
- sync/ordered-async delivery is in the Lab and RGB8 packing is Warp-owned;
- the render request/result vocabulary is implementation-neutral control-plane
  code, not a permanent dual-backend registry.

Every inventoried path and named symbol is checked by a CPU unit test. The test
also rejects production imports of `benchmarks` from the current production
package roots.

## Two-Terminal Cleanup Ledger

`cleanup_ledger.json` assigns a terminal action to both `cuda_selected` and
`warp_selected` for every current backend-specific or comparison surface. All
entries must resolve in P12.4e.

The ledger explicitly covers:

- both executors and Warp optical scene/BVH ownership;
- experimental CUDA LBVH/source ownership and its independent retention gate;
- Lab selectors/fallback, delivery/RGB8, exports, examples, tests, and docs;
- the paired comparison infrastructure;
- any public physics CUDA-binding API introduced solely for the CUDA consumer.

`collapse` means retain only the implementation-neutral or selected-backend
part of a mixed file; it is not permission to leave dormant losing branches.
There is no terminal dual-backend or automatic-fallback state.

## Canonical Cases

`benchmarks/optical/cases.py` is dependency-light and defines five logical
workload families with stable SHA-256 hashes:

1. asymmetric 7x3 direct ray/correctness fixture;
2. Go2 static 960x640 historical anchor;
3. Go2 static 1920x1080 product target;
4. deterministic dynamic rigid replay at 960x640;
5. real physics-published body-triangle shipping case at 960x640.

The definitions freeze camera semantics, transform sequence, lighting source,
output profiles, shadow modes, row-major ray layout, dtype, tie behavior, and
required asset identity. Future lanes must materialize independent Warp arrays
and Torch CUDA tensors from these logical values; neither lane may derive timed
input from the other.

## Frozen Gate and Boundary Schema

`benchmarks/results/q54_p12_4/gate.json` is frozen before new comparison
results. `benchmarks/optical/schema.py` validates it and binds every raw sample
to the gate's canonical semantic SHA-256.

The gate includes:

- versioned R0/R1/R2/D/I/S definitions;
- a normalized R0 topology/order fixture and per-run topology hash check;
- exactly the supported six R2 payload/policy combinations, without invented
  full async support;
- the same six payload/policy families for shipping S, with distinct host-ready,
  consumer-completion-recorded, logical-slot-reclaimable, final-drain, and
  final-reclaim endpoints;
- canonical full readback derived from the implementation-neutral optical output contract,
  including per-ray `material_index` and the four primary scalar diagnostics;
- explicit per-ray/scalar channel names, dtype, shape, individual bytes, and
  total transferred bytes for none/RGB/full/RGB8 payloads;
- lane-produced, content-hashed execution receipts plus row-semantics hashes and explicit renderer, acceleration, profile, shadow,
  payload, ring-depth, observed-channel, completion, and lifecycle evidence;
- unprofiled primary passes, separate diagnostic profiling, alternating paired
  order, independent shipping resets, warmup/sample/repeat counts, and empty,
  event, and copy calibration rows;
- `CUDA/Warp <= 1.00` on every mandatory matched row, a 3 percent noise band,
  bounded reruns/optimization slices, deadline, hard failures, and default Warp
  cleanup when unresolved;
- a 30 FPS shipping latency/throughput floor;
- clean single-lane NVML peak-memory measurement from before physics
  initialization through final reclaim, with no numeric memory pass threshold.

The memory process contains physics, one candidate optical backend, that
candidate's acceleration, delivery/readback rings, and allocator-reserved
memory. It may not load the losing lane, subtract a baseline, or clear caches to
manufacture a lower peak. OOM on a mandatory workload remains a hard failure.
A numeric requirement is added only by a future owner decision tied to a named
product workload and deployment profile before its measurements are inspected.

## Review Corrections Before Baseline

Claude's implementation reviews found contract and provenance holes, all
corrected before a runner or baseline exists:

1. The GPU device `DIRECT_LIGHT_FULL` path actually returns and stages
   `material_index`. Primary delivery also transfers four scalar diagnostics.
   `material_index` is now part of the formal `DIRECT_LIGHT_FULL` profile and is
   emitted by the CPU path using the registry ordinal shared with device scene
   construction.
2. A gate hash alone did not prove a sample executed its row. The result schema
   carries the row-semantics hash and actual execution/payload/lifecycle receipt.
   Validation rejects wrong renderer/acceleration, output profile, shadow mode,
   payload/policy, ring depth, channel set, dtype/shape/bytes, completion point,
   drain, or reclaim evidence.
3. `optics/output_contract.py` is the sole CPU-safe source for profile-required
   channels and none/RGB/RGB8/full host dtypes, shapes, and byte calculations.
   Executors, staging, Lab delivery, and benchmark schema all consume it.
4. Result schema v3 freezes an `ExecutionReceipt` produced by the backend lane.
   Its content hash excludes coordinator metadata; the coordinator may attach
   only run/repeat/sample identity and timing, never execution, payload,
   lifecycle, counter, or memory observations.
5. Strict FULL delivery initially filtered out the optional nine shadow-
   traversal counters. They now form the explicit
   `shadow_traversal_counters_v1` diagnostic overlay: nine `[1]` `int32`
   scalars, 36 bytes total. Host and device FULL delivery merge that overlay
   only when traversal diagnostics are requested. Primary FULL payload bytes
   and replacement gates remain unchanged.

## C0 Lifecycle Preconditions Frozen by a1

The gate incorporates the reviewer's non-blocking state-machine suggestions as
blocking C0 acceptance inputs:

- duplicate completion, completion without borrow, and out-of-order completion
  are rejected;
- close/unbind with an active lease is rejected;
- destroyed or reused stream handles fail closed without logical completion;
- ordinary exceptions record a valid done event after already-enqueued readers,
  then advance logical completion;
- event-record failure and CUDA-fatal failure poison the consumer/slot and abort;
- logical slot reclaimability is never reported as physical GPU completion;
- host polling/synchronization, private ring mutation, Torch event insertion,
  and a Warp-optical adapter are forbidden.

This slice only freezes those acceptance rules. It does not design or implement
the public binding; that remains P12.4c0a/c0b.

## Repository Hygiene

Only the reviewed gate, inventory, and cleanup ledger are tracked under
`benchmarks/results/q54_p12_4/`. Raw samples and generated result artifacts are
ignored by default and must be referenced by stable location plus SHA-256.

The durable dependency direction is tested:

```text
benchmarks -> production modules     allowed
production modules -> benchmarks     forbidden
```

No benchmark result, copied kernel, product fallback, or external package was
added. `PROGRESS.md`, `REFLECTIONS.md`, and `MANIFEST.md` record the approved
milestone and next slice.

## Files Added

- `benchmarks/optical/__init__.py`
- `benchmarks/optical/cases.py`
- `benchmarks/optical/schema.py`
- `optics/output_contract.py`
- `benchmarks/results/q54_p12_4/gate.json`
- `benchmarks/results/q54_p12_4/architecture_inventory.json`
- `benchmarks/results/q54_p12_4/cleanup_ledger.json`
- `tests/unit/benchmarks/__init__.py`
- `tests/unit/benchmarks/test_optical_benchmark_contracts.py`
- `collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v2.md`
- this implementation note

## Files Updated

- `.gitignore`
- `optics/device.py`
- `optics/execution.py`
- `optics/scene.py`
- `optics/__init__.py`
- `tools/optical_pipeline_lab/delivery.py`
- `PROGRESS.md`
- `REFLECTIONS.md`
- `MANIFEST.md`

## Verification

Targeted contract and render-API tests:

```text
conda run -n robot_sim env PYTHONPATH=. python -m pytest \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py -q

272 passed in 28.10s
```

Full commit gate including GPU tests:

```text
conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/ -m 'not slow' -q

1532 passed, 39 skipped, 117 deselected, 1 pre-existing warning in 302.98s
```

Static and repository checks:

```text
conda run -n robot_sim python -m ruff check \
  optics/output_contract.py optics/device.py optics/execution.py optics/scene.py optics/__init__.py \
  optics/cuda_direct_light.py optics/warp_execution.py \
  tools/optical_pipeline_lab/delivery.py tools/optical_pipeline_lab/timing.py \
  benchmarks/optical \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_direct_light_executor.py \
  tests/unit/optics/test_optical_pipeline_lab.py

All checks passed!
```

`python -m json.tool` passes for all three tracked JSON contracts, all current
inventory paths/symbols resolve, `git diff --check` passes, and the worktree is
clean after the implementation commit.

## Reviewer Focus

1. Confirm the shared output contract drives profile filtering, host staging,
   delivery and gate byte derivation, including `material_index`, four primary
   diagnostics, and the opt-in nine-counter traversal overlay.
2. Confirm result schema v3 requires a valid lane-produced receipt hash and
   cannot accept a row with mismatched execution,
   payload, completion, drain, or reclaim evidence.
3. Confirm removal of a numeric memory threshold and the clean single-lane
   report-only measurement contract match owner decision v2.
4. Confirm the C0 exception/poisoning and ordered-completion rules are the right
   inputs for the later public binding design.
5. Confirm every cleanup entry reaches a genuinely single-backend terminal
   repository and that no mixed file uses `collapse` to preserve a fallback.

## Known Limitations

- There is no executable benchmark runner, lane adapter, comparator, artifact
  writer, or approved baseline in a0/a1.
- The gate is structurally validated, but performance evidence begins only
  after a2/a3 and cannot drive optimization until a4 comparator review.
- The C0 public CUDA publication contract remains unimplemented.
- This initiative has no numeric product-capacity memory claim; adding one
  requires the workload-bound owner decision described in decision v2.
- CUDA remains blocking, Warp-owned for scene/BVH inputs, non-RGB8 in product
  paths, incomplete for plane shadows, and is not the default backend.
- Warp remains the product/oracle path. Nothing is retired in this slice.

## Commit

This note is included in the implementation commit. Its exact hash is reported
in the reviewer handoff so the note does not attempt a self-referential commit
hash.
