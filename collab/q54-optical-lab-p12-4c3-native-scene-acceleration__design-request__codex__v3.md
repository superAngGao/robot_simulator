# Q54 Optical Lab P12.4c3 Native Scene and Acceleration Design Request v3

Author: Codex

Date: 2026-08-17

Status: P1 provenance-DAG/R0-fixture and P2 C3/C4 ownership revision proposed
for owner and Claude re-review; implementation, a3b execution, and baseline
generation remain unauthorized

Supersedes:
`q54-optical-lab-p12-4c3-native-scene-acceleration__design-request__codex__v2.md`
(`8a1dfd2`)

Parent gate: P12.4 proposal v5 (`46a5d33`)

Frozen native-lane correction:
`q54-optical-lab-p12-4a3-native-lane-dependency-order__proposal__codex__v3.md`
(`57073b8`)

Approved prerequisites:

- P12.4a0/a1 benchmark contracts (`231a541`);
- P12.4a2 native-input preparation (`1824fd9`);
- P12.4a3a runtime eligibility instrumentation (`3d0d42a`);
- P12.4c0a/c0b CUDA publication lifecycle (`a1e603b`, `706ab85`);
- P12.4c2 native CUDA execution timeline and fixed diagnostics ABI
  (`3ec732e`, `8325c43`).

## Owner summary

P12.4c3 should now replace the allocating scene/BVH compatibility paths with
prepared, caller-output scene-update and acceleration commands. It must not
turn the comparison into a permanent cross-runtime product abstraction.

The CUDA candidate gets a Torch-owned optical data plane:

```text
immutable logical scene
-> independent Torch CUDA local-scene upload              setup only
-> preallocated world geometry/AABB scene update          timed R1/R2
-> preallocated Morton/Karras LBVH rebuild                timed R1/R2
-> existing C2 CUDA render enqueue                        timed R1/R2
```

The Warp candidate keeps the product strategy already frozen by the gate:

```text
immutable logical scene
-> independent Warp local-scene upload                    setup only
-> CPU median leaf-4 topology upload                      setup only
-> prebound Warp world geometry/AABB update               timed R1
-> prebound Warp leaf/internal BVH refit                  timed R1
-> Warp render                                             later a3b
```

For the frozen static Go2 R2 comparison only, the Warp-renderer lane uses the
same CUDA LBVH algorithm through stable setup-time Torch views of Warp-owned
AABB and LBVH arenas. The timed transaction creates no view, performs no copy,
and uses one underlying CUDA stream. This is comparison infrastructure, not a
general Warp-optical/CUDA-optical bridge.

Only the CPU-safe logical scene packing contract is shared between candidates.
Each runtime owns independent allocations, readiness, commands, and mutable
outputs. P12.4e can therefore remove either losing data plane without leaving
the selected backend dependent on it.

For benchmark lanes, those independent runtime allocations are created once by
the a2 native-input preparation. C3 wraps the exact prepared allocations into
typed local-scene and replay views; it does not perform a second scene upload
that could diverge from the readiness manifest. The packed scene, a2 manifest,
native readiness, runtime local scene, C3 stage evidence, acceleration evidence,
and final receipt all carry one `scene_contract_sha256`.

C3 does not implement C4 leases/arenas, B3 payload delivery, a3b lane
submission, S physics-to-image activation, a baseline, or a backend decision.

## Current stage recap

The repository already has the following reviewed foundations:

1. P12.4a0/a1 freezes workload identities, R0/R1/R2/S boundaries, payload
   semantics, lane-generated receipts, memory reporting, and terminal cleanup.
2. P12.4a2 independently materializes one immutable logical input into Warp-
   owned and Torch-owned CUDA allocations and proves manifest/content parity.
3. P12.4a3a installs runtime hooks around real entry points and keeps every
   result boundary closed until native lanes exist.
4. P12.4c0 closes the Warp-on-CUDA physics producer to external CUDA consumer
   publication lifecycle, including poisoned shutdown and cross-owner resource
   guardians.
5. P12.4c2 supplies an indexed Torch stream, ordered immutable transaction
   stages, allocation-free CUDA raygen/first-hit/shade entries, and exactly one
   owner-controlled optical terminal event.

The remaining blockers before a3b are:

- C3: native scene update and acceleration;
- C4: reusable per-shape/profile slots and their leases/reclaim;
- B3: out-style RGB8 pack and matched sync/async delivery.

The current `update_device_optical_scene_from_gpu_frame(...)` cannot satisfy
C3 because it allocates every world buffer and records a ready event per call.
The current `build_cuda_lbvh_from_snapshot(...)` also synchronizes, creates
Warp/Torch views, invokes allocating Torch reductions and extension entries,
allocates CUB scratch internally, wraps outputs back into Warp, and records a
second ready event. These functions remain compatibility surfaces until the
new commands are validated; they are never relabeled as native benchmark
entries.

## P1 decision: one mandatory input-provenance chain

`PackedOpticalSceneHost` is the only materialization source for optical scene
arrays. Registry traversal, degenerate filtering, primitive order, instance and
material ordinals, source keys, role masks, local geometry, and static optical
metadata are computed exactly once into that object. No Warp builder, Torch
builder, a2 materializer, lane adapter, or benchmark case may independently
repack those scene arrays from a registry.

The frozen chain is:

```text
gate row case_id + case_sha256
-> PackedOpticalSceneHost.scene_contract_sha256
-> CanonicalNativeInput / NativeInputManifest
-> independently prepared Warp or Torch NativeInput allocations
-> NativeReadinessEvidence
-> typed runtime local-scene wrapper over those exact allocations
-> bound scene-update outputs
-> bound acceleration inputs/outputs
-> bound renderer inputs
-> lane-produced stage/acceleration/execution evidence
```

Every arrow is validated mechanically before the first enqueue. Equality of
shape, dtype, counts, registry label, or numerical output is never accepted as
a substitute for the hash and exact-storage links.

Transform provenance is independent from static scene provenance:

- R1/R2 `canonical_native_replay_v1` carries a full
  `transform_replay_sha256` plus the selected `transform_frame_index`;
- R0 carries neither replay field and instead carries the exact
  `fixed_render_fixture_sha256` of its prepared normalized world-scene/BVH
  fixture;
- S carries neither replay field and instead binds the exact C0 publication
  run/step/frame/slot evidence from `ShippingPublicationReceipt`.

A scene hash cannot satisfy a transform hash, and matching transform values
cannot satisfy a scene, replay, or publication identity.

### Schema evolution before any result

C3a upgrades the native-input and receipt contracts before C3b, a3b, or any
baseline is executable. The gate's native-input schema, row-semantics hash
inputs, and receipt schema versions are bumped in the same reviewed commit. No
performance threshold or workload meaning changes.

`NativeInputManifest` adds:

```text
scene_contract_schema_version
scene_contract_sha256
scene_array_names
transform_replay_sha256       # non-null only for replay-backed R1/R2
transform_frame_count         # zero when no replay exists
fixed_render_fixture_sha256   # non-null only for fixed_native_scene_v1 / R0
fixed_render_fixture_arrays   # empty outside R0
```

Its manifest SHA-256 includes these fields. `scene_array_names` identifies
every and only array copied directly from the packed scene and prevents an
additional array with a scene semantic from bypassing the contract.

`NativeReadinessEvidence`, `NativeInputValidationReport`, Warp/Torch prepared
inputs, runtime local-scene wrappers, scene-stage evidence, acceleration-stage
evidence, `AccelerationReceipt`, and `ExecutionReceipt` each add the same
`scene_contract_sha256`. The manifest, validation, readiness, and prepared
inputs carry the full `transform_replay_sha256` and frame count. The selected
transform input, scene-stage evidence, native execution binding, and execution
receipt carry that same replay hash plus the exact per-sample
`transform_frame_index`. The canonical `input_provenance_sha256` is upgraded to
include
`case_sha256`, `scene_contract_sha256`, `transform_replay_sha256`,
`transform_frame_index`, `fixed_render_fixture_sha256`, and
`native_execution_binding_sha256` in addition to the existing
manifest/content/topology/asset/publication identities.

For R0, the manifest, validation, readiness, prepared input, fixed renderer
inputs, native execution binding, acceleration receipt, and execution receipt
all carry the same `fixed_render_fixture_sha256`; their replay fields are
required to be absent. For R1/R2 the fixed-fixture field is required absent.
For S both fixed-fixture and replay fields are required absent and publication
evidence supplies the measured transform identity.

The result-capable lane, after C4 has supplied final slots and sealed stages,
produces immutable `NativeExecutionBindingEvidence` before timing. C3a defines
its schema/canonical validator and exercises it with explicitly non-result-
capable one-shot bindings. It binds:

```text
row case identity
native-input manifest/readiness identity
scene_contract_sha256
runtime local-scene storage manifest
transform replay hash/frame index, fixed-render-fixture hash, or publication identity
ordered stage-evidence map with explicit operation names
dataflow_binding_sha256
```

The stage-evidence map contains only stages that actually execute. R0 contains
`render` only; R1 contains `scene_update`, `acceleration`, and `render`; R2 adds
the later B3 stages; S additionally binds publication evidence outside this C3
slice. Missing R0 scene/acceleration stages are required absences, not null
hashes or fabricated evidence.

`ExecutionReceipt` carries this object as `native_execution_binding` plus its
canonical `native_execution_binding_sha256`. The binding schema is lane-owned;
the runner accepts it as input for validation but never manufactures it from a
row.

### Canonical provenance DAG

The hash dependency is strictly one-way:

```text
PackedOpticalSceneHost / replay or fixed-fixture manifests
-> prepared native storage manifests
-> independent scene / acceleration / renderer stage evidence
-> NativeExecutionBindingEvidence + dataflow_binding_sha256
-> ExecutionReceipt + input_provenance_sha256
```

Every node uses a versioned canonical JSON payload. A node may reference only
hashes from an earlier DAG level. In particular:

- scene-stage evidence records source facts, transform/fixed-fixture identity,
  input/output storage manifests, operation identity, and dispatch plan;
- acceleration-stage evidence records source/topology facts, AABB and BVH
  storage manifests, workspace/operation identity, and dispatch plan;
- renderer-stage evidence records fixed-fixture or scene/BVH input storage
  manifests, output storage manifest, and render operation identity;
- none of those stage-evidence payloads contains
  `native_execution_binding_sha256`, `dataflow_binding_sha256`, receipt hash,
  or a reference to an aggregate binding object;
- only after every required stage evidence is frozen does the lane compute
  `NativeExecutionBindingEvidence` and its dataflow hash;
- only after binding evidence is frozen may the execution receipt reference
  its hash.

`AccelerationReceipt` may reference the independent acceleration-stage
evidence SHA-256, scene contract, topology validation, and actual runtime trace.
It never references native execution binding identity. The enclosing execution
receipt is the first object allowed to contain both acceleration receipt and
native execution binding hashes.

`dataflow_binding_sha256` is derived from the validated storage signatures and
active ranges that prove:

- for R0, renderer inputs are the exact a2 fixed-render-fixture allocations;
- the scene-update stage reads the exact runtime local scene and selected
  replay/publication transform input when that stage exists;
- acceleration reads the exact AABB outputs written by that scene stage when
  acceleration exists;
- render reads the exact fixed-fixture inputs or the world-scene/BVH outputs
  written by the preceding stages;
- no same-shaped substitute storage is present in the measured dataflow.

The single-lane runner validates this binding evidence against the row,
prepared input, readiness, and registered stages before entering
`native_timed_submission(...)`. The resolved execution receipt must contain the
same binding evidence/hash. The coordinator may attach run/repeat/sample timing
metadata but may not construct or overwrite any provenance field.

Any mismatch fails before the first enqueue. There is no warning, fallback,
hash substitution, or post-result acceptance path.

## Frozen identities and timed work

C3 implements the exact gate identities without renaming or weakening them.

| Lane/workload | Scene update | Acceleration | Classification |
|---|---|---|---|
| Warp dynamic replay | `warp_local_scene_transform_and_aabb_out_v1` | `warp_bvh_leaf_internal_refit_out_v1` | refit |
| CUDA dynamic replay | `torch_cuda_local_scene_transform_and_aabb_out_v1` | `cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1` | rebuild |
| Warp static Go2 R2 | `warp_local_scene_transform_and_aabb_out_v1` | `cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1` | rebuild |
| CUDA static Go2 R2 | `torch_cuda_local_scene_transform_and_aabb_out_v1` | `cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1` | rebuild |

The associated acceleration identities remain:

- Warp dynamic: `cpu_median_leaf4_topology_warp_refit_v1`;
- CUDA dynamic: `cuda_morton_karras_lbvh_rebuild_v1`;
- both static Go2 R2 lanes: `cuda_morton_karras_lbvh_rebuild_v1`.

For R1/R2 replay, candidate-native transform sequences are uploaded and the
measured frame view is selected before timing. The timed transaction begins
before scene update. No measured-frame world geometry, AABB, scene bound,
Morton key/order, topology, or node bound may be prepared before that point.
Its exact input contract remains `canonical_native_replay_v1`.

R0 consumes the candidate's independently a2-uploaded
`CanonicalFixedRenderFixtureHost` arrays described below. It performs no C3
scene-update or acceleration operation, but its renderer binding and receipt
remain inside the same exact-allocation provenance chain.

S remains different: it starts before a real `GpuEngine.step(...)`, and its
measured transform view may be created only after C0 borrow inside the timed
handoff. C3 defines scene commands that can consume a validated same-device
Torch transform view, but it does not claim that the C0 view binding and C2
transaction-preparation boundary has been composed for S. P12.4b4 must review
that per-borrow activation without a per-slot view cache before S is enabled.
The S input contract remains `gpu_engine_published_frame_v1`; replay provenance
cannot satisfy it even when transform values are numerically identical.

## Scope and non-goals

### In scope

- one implementation-neutral CPU host scene contract;
- a2 manifest/readiness and receipt-schema evolution that binds that contract;
- independent Warp and Torch CUDA uploads from that contract, performed once
  by a2 preparation and wrapped directly by benchmark C3 local scenes;
- independent replay hash/frame provenance and bind-time scene-to-render
  storage-flow evidence;
- caller-owned Warp and Torch world-scene/AABB outputs;
- prepared, prebound Warp scene-update commands;
- prepared, prebound Warp leaf/internal refit commands over the exact frozen
  CPU topology;
- a Torch CUDA out-style scene-update extension and transaction stage;
- a Torch CUDA out-style Morton/Karras rebuild extension and transaction stage;
- caller-owned scene-bound, Morton, topology, bound, counter, and CUB/reduction
  scratch storage;
- stable setup-time interop for the Warp-renderer/static-CUDA-LBVH comparison;
- direct scene, topology, bounds, and traversal correctness tests;
- runtime instrumentation and immutable execution evidence for the new entry
  points;
- cleanup-ledger coverage for every new production and comparison-only symbol.

### Out of scope

- CUDA BVH refit;
- an alternate builder, leaf-size search, SAH tuning, or fused render kernel;
- general arbitrary-size arena policy;
- optical output/readback lease ownership;
- RGB8 packing, D2H, async delivery, or final drain;
- the per-borrow S transaction adapter or CUDA optical product enablement;
- a new permanent backend registry or general Warp/Torch bridge layer;
- a3b `submit_sample()`, result artifacts, performance numbers, threshold
  changes, baseline inspection, backend selection, or default changes.

## One logical scene contract, two runtime owners

Move the current private registry packing logic into the CPU-safe module
`optics/scene_contract.py`.

The module provides an immutable `PackedOpticalSceneHost` containing the
current canonical arrays and scalar counts:

- local triangle vertices and primitive/instance/material/source metadata;
- local plane point/normal and instance/material/source metadata;
- instance body indices and body-to-geometry transforms;
- material and light arrays;
- the deterministic role-to-mask table;
- primitive, plane, instance, material, light, and maximum-body counts;
- `scene_contract_schema_version`, `source_contract_id`,
  `source_contract_sha256`, `scene_id`, and `materializer_id`;
- a canonical `scene_contract_sha256` over that source identity plus all array
  manifests/content, scalar counts, semantic names, and role-table entries.

Every array is C-contiguous, read-only after construction, and has one formal
dtype/shape. Degenerate-triangle filtering, primitive order, numeric instance
IDs, source-order keys, role masks, and material ordinals retain current
behavior. `optics/scene_contract.py` imports neither Torch nor Warp and never
imports the benchmark package.

Production `build_device_optical_scene(...)` is changed to consume this
contract for its Warp upload. The new production Torch builder consumes the
same contract for a Torch CUDA upload. Neither runtime allocation is a view of
the other.

Benchmark materialization is stricter. Replace the result-capable generic a2
construction path with
`CanonicalNativeInput.from_scene_contract(packed_scene, ...)`. That constructor
inserts the packed arrays by object identity, rejects name collisions or scene-
semantic arrays in additional inputs, and requires
`packed_scene.source_contract_id/source_contract_sha256` to equal the canonical
case ID/hash used by the manifest. It also requires `packed_scene.scene_id` to
equal the case's frozen scene ID and binds the exact materializer ID. It then
computes the manifest fields above.
The generic `from_arrays(...)` constructor remains only for isolated unit
fixtures and produces `scene_contract_sha256=None`; such an input is rejected
by every a3b row and can never receive result-capable readiness.

`prepare_warp_native_input(...)` and `prepare_cuda_native_input(...)` perform
the one independent upload per candidate. C3 benchmark local-scene constructors
accept only the resulting `WarpPreparedNativeInput` or
`CudaPreparedNativeInput`. They select the manifest-declared scene arrays by
exact name and wrap those exact allocations without allocation, copy, upload,
or cross-runtime view creation. They retain the prepared input and manifest as
strong owners. A separately uploaded scene, even from an equal registry, is
not accepted by a benchmark lane.

Tests compare both independently prepared native allocations element by
element against the same packed host contract after readiness is resolved
outside timing.

This shared contract is durable under either terminal decision. If CUDA wins,
the Warp upload/data plane can be deleted while the host packer remains. If
Warp wins, the Torch upload/data plane can be deleted without moving canonical
packing back into a Warp module.

### Separate replay contract

R1/R2 replay transforms use an immutable `CanonicalTransformReplayHost` in
`benchmarks/optical/native.py`, not ad-hoc additional arrays. It carries its
own schema version, canonical case
ID/hash, transform-sequence semantics, frame count, `x_world_R`/`x_world_r`
array manifests/content, and `transform_replay_sha256`.

`CanonicalNativeInput.from_scene_contract(...)` requires this replay object
for `canonical_native_replay_v1`, inserts its arrays directly, and rejects
transform-semantic arrays supplied through another argument. It rejects a
replay whose case ID/hash differs from either the packed scene or native-input
case. R0 and S reject a replay object.

The two a2 candidate uploads independently copy the full replay once. A
replay-backed C3 transform input is only a typed view of frame
`transform_frame_index` within that exact prepared allocation. Bounds checking
and storage-signature validation happen before timing. S instead constructs its
measured transform view from the C0-borrowed physics slot and must not carry a
replay hash/index.

### Formal R0 fixed-render fixture

`fixed_native_scene_v1` is not allowed to bypass the packed-scene/a2 chain.
C3a adds immutable `CanonicalFixedRenderFixtureHost` in
`benchmarks/optical/native.py`, constructed only from:

- one `PackedOpticalSceneHost` already bound to the canonical case;
- the case's canonical fixed world transform;
- the frozen CPU median/longest-axis leaf-4 normalized BVH builder and source
  order used by R0.

It contains every array consumed by the R0 renderer:

- world triangle v0/e1/e2/normal and plane point/normal;
- primitive/instance/material/source metadata and shade material/light arrays;
- normalized BVH bounds, left/right/start/count, and primitive IDs;
- active primitive/plane/node counts and sensor-role/render scalars that are
  part of input identity.

Its canonical `fixed_render_fixture_sha256` includes the case ID/hash,
`scene_contract_sha256`, fixture schema/materializer IDs, fixed transform
identity, every world-scene and BVH array manifest/content hash, normalized
topology hash, active counts, and semantic names. It cannot be constructed from
independently supplied world arrays or a pre-existing backend scene/BVH.

For `fixed_native_scene_v1`,
`CanonicalNativeInput.from_scene_contract(...)` requires this fixture, verifies
that it was derived from the exact packed object/case, inserts its arrays by
object identity, and records `fixed_render_fixture_sha256` plus every fixture
array name in `NativeInputManifest`. Replay input is forbidden for this branch.
For replay rows the fixture is forbidden; S accepts neither replay nor a fixed
render fixture for its measured transforms.

a2 independently uploads the complete fixture into Warp-owned and Torch-owned
allocations. R0 renderer bindings wrap those exact prepared allocations; they
may not use `build_device_optical_scene(...)`, a CUDA compatibility upload, a
separately allocated normalized BVH, or same-shaped substitute buffers.

R0 `NativeExecutionBindingEvidence` contains the fixture hash, prepared-storage
manifest, renderer-stage evidence, renderer input/output storage signatures,
and a dataflow hash proving exact fixture-to-render storage identity. It has no
scene-update or acceleration-stage evidence and must not invent either stage.
Its `AccelerationReceipt` remains classification `none`, with zero scene/refit/
rebuild and kernel counts, while topology validation is bound to the fixture's
normalized topology hash.

## Runtime-owned structures

The implementation should expose narrow immutable bundles rather than a
polymorphic cross-backend scene object.

### Torch CUDA ownership

Conceptual structures are:

```text
CudaOpticalLocalScene
  immutable typed wrapper over the exact Torch a2 local geometry, metadata,
  material, and light allocations plus their scene contract/manifest owner

CudaTransformInputs
  same-device contiguous x_world_R / x_world_r plus env index and a strong
  source owner, transform_replay_sha256, and frame index; for S the C0 lease
  and publication identity replace replay provenance

CudaWorldSceneBuffers
  caller-owned triangle v0/e1/e2/normal/AABB and plane normal/point outputs

CudaLbvhBuffers
  caller-owned scene bounds, Morton input/sorted keys, primitive ids,
  left/right/parent/start/count/range/split, node bounds, and child counters

CudaLbvhWorkspace
  caller-owned opaque aligned reduction/radix-sort workspace with a frozen
  extension/device/primitive-count requirement manifest
```

The new world/BVH structures are active views over caller-owned storage. C3
adds `allocate_cuda_world_scene_buffers(...)` and
`allocate_cuda_lbvh_buffers(...)` only for tests and compatibility callers.
Their documentation states that they execute before submission and do not
satisfy C4.

### Warp ownership

The existing `DeviceOpticalScene` remains the production Warp local-scene
owner. Its benchmark construction wraps the exact a2 Warp arrays and retains
their manifest/readiness owner. Add
caller-owned world-scene/AABB buffers and a prepared update plan rather than
allocating a fresh `DeviceOpticalSceneSnapshot` in the measured call.

The dynamic Warp BVH uses:

- one CPU median/longest-axis leaf-4 topology created from the canonical setup
  frame outside timing;
- one topology upload per prepared lane/session;
- immutable left/right/start/count/primitive-order/depth/level-range buffers;
- caller-owned mutable node-bound buffers;
- a prepared leaf-refit launch followed by prepared internal-level launches in
  deepest-to-root order.

No C3 API mutates the immutable topology or silently rebuilds it.

## Prepared Warp commands

Warp 1.12 supports `wp.launch(..., record_cmd=True)`, which constructs a
`wp.Launch` without submitting it. C3 uses that facility during preparation:

- C3 delivers `WarpSceneUpdateCommandFactory` and
  `WarpBvhRefitCommandFactory` protocols that own compiled kernels, immutable
  local-scene/topology facts, and validation logic but no final C4 output slot;
- a factory accepts one concrete transform-frame view, output storage bundle,
  BVH storage bundle, and prepared stream, then records and seals one immutable
  command bundle outside timing;
- arguments, dimensions, device, exact input/output storage, transform-frame
  view, and launch order are frozen in that sealed bundle;
- timed enqueue calls only `Launch.launch(stream=prepared_stream)` in the
  frozen order;
- no timed `wp.zeros`, `wp.array`, argument list construction, event creation,
  event wait, ready-event record, host query, or host/global synchronization is
  allowed.

C3 validates the factory/protocol with explicit one-shot caller-owned test
buffers. Those buffers and bundles are not C4 arena slots, carry no product
slot/generation/ring-depth claim, and cannot make a lane result-capable.

C4 later allocates the final scene/BVH/output slots and fixes ring depth. For
every required `(replay_frame_index, output_slot_id)` pair, C4 calls the C3
factory once with the final storage and seals a new bundle. A sealed bundle is
never rebound. Each C4 lease supplies the current `slot_id` and `generation`
and validates them against the bundle's immutable slot/storage signature before
enqueue. Reuse changes the lease generation, not the recorded storage pointer.
The sealed bundle/binding hash contains stable slot/storage identity, not a
future lease generation. C4's per-sample lifecycle evidence records the actual
generation and binds it to the submitted bundle/slot; the execution receipt
cross-checks both without mutating stage evidence.
Resize, storage replacement, device/profile change, or arena recreation
invalidates and destroys the old bundle set; C4 creates and seals a new set
before the replacement arena becomes available.

The scene plan contains at most one triangle command and one plane command.
The refit plan contains one leaf command and the exact immutable sequence of
non-empty internal-depth commands. Empty primitive classes enqueue no dummy
kernel. Command count and order become immutable stage evidence.

Prepared commands and the selected Warp stream are strongly owned. Factory
seal and C4 preflight validate the device, native stream handle, array pointer,
dtype, shape, contiguity, capacity, topology hash, and output disjointness. A
Warp version/API change that invalidates recorded-command assumptions fails
preflight and requires review; it does not fall back to per-frame `wp.launch`
argument construction.

These prepared commands do not depend on `optics/cuda_timeline.py`. The
temporary benchmark lane owns Warp completion and later delivery orchestration,
so a Warp-selected cleanup does not inherit the CUDA-candidate transaction
module.

## Torch CUDA scene-update stage

Add `optics/cuda_scene.py` with a versioned extension prepared outside timing.
It exports two fixed out-style entries selected and bound outside timing:

- `update_triangles_world_aabb_out(...)`;
- `update_planes_world_out(...)`.

Both entries obey the same rules:

- all Tensor arguments are explicit fixed-position parameters;
- no Python sequence is converted to `std::vector<torch::Tensor>`;
- no Torch/CUDA/event/scratch allocation occurs in C++;
- all input/output tensors are validated for the exact indexed CUDA device,
  dtype, shape, contiguity, capacity, and non-overlap;
- an output tensor is the device/stream anchor;
- C++ installs `c10::cuda::CUDAGuard` and obtains
  `at::cuda::getCurrentCUDAStream(anchor.get_device())`;
- triangle and plane kernels use that one stream and issue launch checks only;
- every active output element is overwritten, including world AABBs;
- empty triangle or plane sets do not require dummy storage or a dummy launch;
- there is no host wait, default-stream assumption, or ready-event record.

`CudaSceneUpdateStageFactory` produces a stage implementing the C2
transaction-stage protocol and named `cuda_scene_update`. Factory sealing
freezes both exact extension callables and their
argument tuples, transform input, local scene, world output, environment index,
active counts, storage signatures, native-input manifest hash,
`scene_contract_sha256`, `transform_replay_sha256`, and selected replay frame
index.
`enqueue(transaction)` enters `transaction.enqueue_stage(self)`, records
optional diagnostic phase events, and invokes only the non-empty already-bound
entries.

For R1/R2 replay, the transform view and complete stage are bound before the
timer. A later S adapter may supply a C0-borrowed zero-copy transform view, but
C3 neither caches such views by physics slot nor changes the S start point.

## Native CUDA Morton/Karras rebuild

Move the production extension source out of
`examples/cuda_lbvh_extension_spike.py`. During migration, the example becomes
a thin caller of the production API. Production never imports CUDA source from
`examples/`.

The initial native identity is exactly:

```text
cuda_morton_karras_lbvh_rebuild_v1
topology_origin = cuda_morton30_karras_source_index_tiebreak_v1
topology_lifetime = transaction_output_arena_lease_v1
```

The ordered timed operation is:

```text
world triangle AABBs
-> scene bounds
-> 30-bit Morton code plus source-index composite key
-> deterministic CUB radix sort
-> Karras topology
-> leaf bounds
-> internal bounds
```

The composite key remains `(morton30 << 32) | source_index`, so equal Morton
codes are ordered by canonical source index. C3 does not call this a refit.

### Workspace query and fixed ABI

The extension exposes
`cuda_lbvh_workspace_requirements(primitive_count, device_ordinal)` outside
timing, keyed by:

- extension ABI/source digest;
- target device ordinal and compute capability;
- primitive count and derived node capacity;
- bounds-reduction and CUB radix-sort requirements.

The query uses CUB's null-workspace sizing protocol outside timing. The caller
then allocates the required opaque workspace and every typed input, output, and
counter tensor before bind. Bind rejects a missing, undersized, misaligned,
wrong-device, overlapping, or stale workspace manifest.

The timed `cuda_lbvh_rebuild_out(...)` entry has a fixed positional ABI. It
accepts the AABBs, workspace, scene-bound outputs, Morton/key/id buffers,
topology arrays, node bounds, and child-counter storage explicitly. It must not
contain or call:

- `torch::empty`, `zeros`, `full`, `TensorOptions`, or allocating reductions;
- `std::vector<torch::Tensor>` or another dynamic Tensor container;
- `torch.min`, `torch.max`, `torch.sort`, or an allocating fallback;
- extension loading, workspace sizing, event creation, host query/wait, or
  target-wide synchronization.

Same-stream `cudaMemsetAsync` resets zero counters and integer arrays whose
sentinel has a valid byte-fill representation; one explicit initialization
kernel writes floating-point infinities and any remaining typed sentinels. All
kernels and CUB operations use the indexed current stream after a device guard.
CUB workspace is caller-owned and is not lazily replaced.

`CudaLbvhRebuildStageFactory` produces a stage implementing the C2 transaction
protocol as `cuda_lbvh_rebuild`. Factory sealing freezes exact
callables/arguments, device/stream, AABB inputs, every
output, workspace manifest, primitive/node counts, topology identity, and
storage disjointness. It enqueues no terminal event; C2 `finish()` remains the
only optical terminal record after the final registered stage.

Zero primitives produce an empty active BVH without invalid negative
capacities or a CUB call. One primitive produces one leaf/root with no internal
topology launch. Mandatory benchmark rows contain triangles and must still
report positive scene-update and acceleration submission counts.

The allocating `build_cuda_lbvh_from_snapshot(...)` temporarily adapts to the
new prepared implementation outside timing to preserve existing product tests,
but it remains explicitly benchmark-ineligible until cleanup. It does not
remain as the CUDA-selected product entry after C4/B3 migration.

### C3 factory versus C4 arena ownership

The Warp factories above, `CudaSceneUpdateStageFactory`, and
`CudaLbvhRebuildStageFactory` are the C3 deliverables. C3 proves that each
factory can validate concrete caller-owned storage, create a sealed immutable
stage/bundle, enqueue correct work, reject mutation/rebinding, and expose
independent stage evidence. C3 uses only explicit one-shot test/compatibility
storage for that proof.

C3 does not choose final slot IDs, generations, ring depth, reuse policy, or
lease/reclaim state and does not claim that its one-shot bundles are a3b
bindings. The one-shot allocation helpers never appear in result-capable
`NativeExecutionBindingEvidence`.

C4 owns the final arena topology. After allocating all final storage, C4 invokes
the C3 factories to create and seal the exact per-frame/per-slot stage sets used
by a3b. C4 also binds the existing C2 renderer to those same final scene/BVH/
output slots and later binds B3 to the same output lease. It combines those
independent stage hashes to construct result-capable native execution binding
evidence. The pre-enqueue runner validates the C4 lease's slot/generation
against that sealed storage graph.

No factory has a rebind method. Replacement storage requires a new stage or
command bundle and a new evidence/dataflow hash before the slot is published.
This is a lifecycle replacement, not mutation of a sealed object.

## Static Warp-renderer plus CUDA-LBVH binding

The gate deliberately assigns `cuda_morton_karras_lbvh_rebuild_v1` to both
static Go2 R2 lanes. The Warp comparison binding is therefore narrowly
specified:

1. Allocate world AABB and LBVH input/output storage as Warp-owned arrays.
2. Create all Torch views with `wp.to_torch(...)` once during setup.
3. Keep the Warp arrays, Torch view objects, Warp stream, and Torch external-
   stream wrapper strongly owned for the lane lifetime.
4. Require `torch.cuda.ExternalStream(warp_stream.cuda_stream, device=...)`
   and the Warp stream to expose the same nonzero native handle and device.
5. Bind the out-style CUDA LBVH entry to those stable Torch views outside
   timing.
6. Enqueue Warp scene update, CUDA LBVH rebuild, Warp render, and later B3 work
   in order on that one underlying stream.

No timed call may invoke `wp.to_torch`, `wp.from_torch`, DLPack, `.contiguous`,
copy, upload, event bridge, or host/global synchronization. The stage receipt
therefore reports `bridge_creation_count=0`; setup provenance separately says
that stable comparison views exist.

This binding lives in `benchmarks/optical/lanes/warp.py` as temporary comparison
infrastructure. The core CUDA LBVH command accepts Torch tensors and knows
nothing about Warp. The Warp optical renderer knows nothing about the CUDA
candidate scene. If Warp wins, the binding is removed unless the independently
frozen CUDA-LBVH benefit/maintenance gate retains and productionizes that
acceleration; if CUDA wins, the entire Warp binding is deleted.

## Transaction ordering and completion

The CUDA candidate transaction is exactly:

```text
validated stages and caller-owned buffers prepared before timing
-> cuda_scene_update.enqueue(transaction)
-> cuda_lbvh_rebuild.enqueue(transaction)
-> cuda_direct_light_render.enqueue(transaction)
-> later B3 delivery.enqueue(transaction), when applicable
-> transaction.finish() exactly once
```

C3 records no ready or terminal event between scene, acceleration, and render.
Optional profiling events remain typed diagnostics and are absent from primary
runs. `CudaOpticalCompletion.resource_owners` retains the local scene,
transforms, world outputs, LBVH outputs/workspace, render inputs/outputs, and
stage objects until C4 assumes lifecycle ownership.

For a future S transaction, the C0 physics done event is recorded immediately
after the last scene-update reader has been enqueued. It ends only the
physics-input lease. Acceleration, render, delivery, and the optical terminal
event continue independently. C3 does not insert the C0 event into the LBVH or
optical completion API.

The Warp lane uses its own native command sequence and completion owner. C3
does not make the retained Warp path depend on the CUDA-candidate transaction
module merely to make the temporary comparator symmetrical. Both lanes are
still measured from their frozen native-ready inputs to their row-specific
terminal endpoints; runtime-specific host submission cost remains inside that
measurement.

## Validation, failure, and reuse boundary

All structural validation happens before the first enqueue for R0-R2:

- indexed device and exact stream/native-handle identity;
- extension/module and entry identity;
- input/output dtype, rank, shape, contiguity, capacity, pointer, and storage
  generation;
- transform/body/environment bounds;
- primitive/node/leaf counts and integer range;
- complete output and scratch coverage;
- required non-overlap and allowed stable-view alias provenance;
- gate case ID/hash, native-input manifest/readiness hash,
  `scene_contract_sha256`, and runtime local-scene hash;
- replay hash/frame index for R1/R2, or explicit absence plus the required
  fixed/publication provenance for R0/S;
- exact scene-output -> acceleration-input -> render-input storage links and
  their `dataflow_binding_sha256`;
- topology and logical-scene hashes;
- exact stage order and operation identity.

Validation failure enqueues nothing. After the first enqueue, any immediate
scene/refit/rebuild failure poisons the CUDA transaction or invalidates the Warp
submission. No completion or receipt may claim success. C3 has no authority to
reuse affected storage; tests use one-shot owners, and C4 must later define the
normal and poisoned slot disposition.

Asynchronous CUDA failure observed only at terminal resolution also invalidates
the sample and poisons the associated C4 slot once C4 exists. A successfully
recorded terminal event proves ordering, not kernel success or physical
completion.

## Evidence and receipt provenance

C3 adds immutable lane-produced stage evidence. It is not copied from
`gate.json` and cannot be populated by the coordinator.

Scene evidence includes at least:

- exact scene-update operation ID and extension/Warp kernel identity;
- native-input manifest/readiness identity and `scene_contract_sha256`;
- runtime owner, device, stream token, transform-input contract, and transform
  replay hash/frame index or explicit non-replay provenance token;
- primitive/plane/instance counts;
- output storage manifest hash;
- actual prepared command/dispatch count.

Acceleration evidence includes at least:

- the same `scene_contract_sha256` and its independent acceleration-stage
  evidence SHA-256;
- acceleration ID, topology origin/lifetime/hash, and classification;
- primitive/node/leaf counts;
- exact acceleration entry and prepared command/dispatch plan hash;
- workspace requirement/coverage hash where applicable;
- output storage manifest hash.

The later lane adapter derives `AccelerationReceipt` from this evidence plus
the runtime operation trace. It must not copy row expectations back as observed
values. In particular:

- dynamic Warp reports one scene update, one refit, zero rebuilds;
- CUDA LBVH rows report one scene update, zero refits, one rebuild;
- R0 reports zero scene updates/refits/rebuilds;
- mandatory R1/R2 rows report positive scene and acceleration submission
  counts;
- device allocation, host materialization/upload, D2H, global synchronize, and
  timed bridge-creation counts remain zero;
- `completion_event_recorded` becomes true only from the complete transaction
  endpoint, never from a C3 stage-local event.

`ExecutionReceipt.scene_contract_sha256`, its nested readiness and acceleration
fields, and `NativeExecutionBindingEvidence.scene_contract_sha256` must all be
identical. For replay rows, the receipt, binding evidence, transform input, and
scene-stage evidence must also agree on `transform_replay_sha256` and
`transform_frame_index`. The runner recomputes `input_provenance_sha256` from
these actual fields; a lane-supplied digest cannot override a mismatch.

`acceleration_kernel_launch_count` records only explicit kernels launched by
the production source. It does not invent a count for implementation-private
CUB kernels. The named CUB radix-sort invocation is recorded separately in the
reviewed dispatch-plan hash and extension source/ABI digest. If exact hardware
kernel counts are later needed, they belong to a separate profiling artifact,
not the correctness receipt.

## Runtime instrumentation

Update the a3a source manifest and runtime-hook plans to name the real prepared
scene/refit/rebuild enqueue entries. Source hashes remain coverage only.

The existing low-level hooks continue to reject allocation, materialization,
upload, synchronization, bridge creation, staging, and extension loading in a
timed boundary. C3 adds backend-specific extension-loader hook targets where
needed. Native-source ABI tests close what Python hooks cannot observe:

- no allocating Torch API in the CUDA scene/LBVH timed sources;
- no dynamic Tensor container in the pybind ABI;
- no internal CUB/reduction workspace allocation;
- fixed current-device/current-stream launch pattern;
- no hidden event or synchronization call.

Negative tests must show that calling the old scene snapshot or CUDA LBVH
compatibility entry inside timing fails at its first real contaminating
operation, even when the high-level caller does not self-report contamination.
Positive real-GPU tests must execute diagnostics-off and diagnostics-on render
transactions with C3 stages under the installed runtime hooks and observe only
the expected high-level operations.

C3 does not add any executable boundary to `phase_authorization` and does not
add `submit_sample()` to a lane.

## Correctness requirements

Correctness is resolved outside primary timing after terminal completion.

### R0 fixed-fixture parity

For each backend, compare every prepared R0 world-scene, metadata, material,
light, normalized BVH topology/order, bound, and active-count allocation
directly to `CanonicalFixedRenderFixtureHost`. Prove Warp/Torch contents are
equal but allocations are distinct, renderer inputs are exact views of those
prepared allocations, and the fixture/topology hashes match readiness and R0
binding evidence.

### Scene parity

For asymmetric static and dynamic fixtures, compare every element of:

- world triangle v0/e1/e2/normal;
- triangle AABB min/max;
- world plane normal/point;
- all primitive/instance/material/source metadata consumed by traversal and
  shade.

Compare Warp and CUDA independently against a float64/host reference, then
against each other with frozen tolerances. Include non-identity body and
geometry rotations/translations, multiple environments/bodies, static
instances, non-square geometry, empty planes, empty triangles, and one
triangle.

### Warp topology/refit parity

- prove the uploaded topology exactly matches the CPU median longest-axis
  leaf-4 source-order reference;
- recompute every leaf and internal bound on the host for multiple replay
  frames and compare every node;
- prove topology arrays and topology hash never change across refits;
- prove each primitive occurs exactly once and all parent/child/range
  relationships are valid.

### CUDA LBVH parity

- compare composite Morton keys and sorted primitive IDs to an independent CPU
  reference for ties, degenerate scene extents, and ordinary scenes;
- validate one connected acyclic tree with one root, reciprocal parent/child
  links, valid Karras ranges/splits, and exactly one leaf per primitive;
- compare every leaf/internal bound to an independent descendant reduction;
- prove deterministic byte-identical topology for repeated identical input;
- prove rebuild output changes correctly across replay transforms and that no
  stale counter/bound survives output-slot reuse simulations.

### Traversal parity

Bind the resulting scene/BVH directly into each renderer and compare hit mask,
range, position, normal, instance/material/source identity, RGB, and traversal
diagnostics on asymmetric rays/cameras. Include shadows on/off and all three
current direct-light output profiles. Topology may differ between Warp refit
and CUDA rebuild, but first-hit/shade results must not.

### Provenance-negative coverage

Before any GPU enqueue, tests must reject:

- case A a2 prepared input/readiness paired with a case B packed scene whose
  dtype, shape, primitive count, and topology count are deliberately identical;
- the correct a2 manifest paired with a separately uploaded runtime scene,
  including byte-equal storage with a different unbound owner;
- R0 readiness for the correct fixed fixture paired with separately allocated
  world-scene/BVH storage, including byte-equal same-shape storage;
- a case-A R0 fixed fixture paired with a case-B fixture having identical
  primitive/node counts and tensor shapes;
- a correct scene contract paired with the wrong replay hash or wrong replay
  frame index;
- scene-stage evidence whose scene hash differs from readiness;
- acceleration evidence whose scene hash differs from the scene stage;
- acceleration AABB inputs that do not alias the exact active scene-stage AABB
  outputs;
- renderer world/BVH inputs that do not alias the exact active C3 outputs;
- an execution receipt that copies the expected row hashes while its nested
  binding/readiness/stage evidence identifies different inputs;
- replay provenance in S or publication provenance used to satisfy an R1/R2
  replay row.

At least one same-shape/count case-A/case-B mismatch is exercised for each
backend and for both the R0 fixture branch and the replay scene branch. These
are pre-enqueue failures, so the tests also assert zero stage launches and no
completion-event record.

## Implementation slices and review gates

Implementation proceeds in reviewable commits. No later slice may weaken an
earlier acceptance rule.

### C3a — logical scene contract and independent ownership

- extract the CPU-safe packer;
- add `CanonicalNativeInput.from_scene_contract(...)` and retire generic
  construction from every result-capable path;
- add the separately hashed `CanonicalTransformReplayHost` and require it only
  for `canonical_native_replay_v1`;
- add `CanonicalFixedRenderFixtureHost` and require it only for
  `fixed_native_scene_v1`/R0;
- upgrade native manifest/readiness, acceleration/execution receipts,
  `input_provenance_sha256`, row-semantics inputs, and gate schema versions with
  the explicit scene/replay/fixed-fixture fields;
- migrate the production Warp builder without changing outputs;
- make benchmark Warp/Torch local scenes wrap the exact independently prepared
  a2 allocations rather than upload again;
- add `NativeExecutionBindingEvidence` and the pre-timing runner validator;
- add manifest/content/layout parity, CPU-only import, case-mismatch,
  replay-index, R0 wrong-storage, provenance-DAG, and dataflow-storage negative
  tests.

Gate: no runtime bridge or benchmark import in production scene modules, and no
result-capable readiness/enqueue can exist without the complete scene/replay
provenance chain.

### C3b — prebound Warp scene update and refit

- add caller-output Warp world-scene buffers;
- add Warp scene-update/refit command factories and validate sealed one-shot
  command bundles;
- preserve allocating compatibility facades outside native timing;
- add direct scene/topology/refit and runtime-hook tests.

Gate: no timed allocation, list-built launch, event, wait, or synchronization;
no one-shot bundle claims a C4 slot/generation or result eligibility.

### C3c — Torch CUDA scene update

- add the out-style extension, prepared native API, transaction stage, and
  immutable stage factory/evidence;
- compose scene update with C2 render on one context/stream;
- add source ABI, device restoration, failure, mutation, overlap, empty-shape,
  and numerical parity tests.

Gate: scene and render use one transaction and only owner `finish()` records
terminal completion.

### C3d — native CUDA LBVH rebuild

- move production source out of `examples/`;
- add workspace query/manifest and fixed out-style rebuild ABI;
- add the prepared transaction-stage factory and independent evidence;
- adapt the old builder only as an ineligible compatibility facade;
- add deterministic topology/bounds/traversal and contamination tests.

Gate: scene → rebuild → C2 render completes on one stream with caller-owned
storage and no intermediate event/wait.

### C3e — static Warp comparison binding and closure

- validate the stable Warp-owned/Torch-view binding factory for the frozen
  static R2 acceleration axis on one-shot storage; C4 creates final-slot views
  and sealed bundles after arena allocation;
- prove one native stream handle and zero timed bridge creation;
- update a3a manifests/hooks and the cleanup ledger;
- run the focused/full unit and GPU regression suites;
- write an implementation note for Claude review.

Gate: a3b remains closed; no performance sample or baseline is emitted.

## Cleanup contract

C3 extends the mandatory P12.4e ledger before implementation acceptance.

The ledger receives these exact coverage changes:

- a new `logical_optical_scene_contract` entry for
  `optics/scene_contract.py` and its CPU tests, retained in both branches;
- the durable benchmark-core entry adds `CanonicalTransformReplayHost`,
  `CanonicalFixedRenderFixtureHost`, `NativeExecutionBindingEvidence`, their
  schema/validator tests, and selected-backend regression ownership;
- the existing `warp_scene_and_bvh_data_plane` entry adds every prepared Warp
  scene/refit symbol and dedicated GPU test;
- a new `cuda_native_scene_data_plane` entry covers `optics/cuda_scene.py`, its
  extension cache identity, exports, and dedicated unit/GPU tests; it is
  retained only for CUDA selection;
- the existing `cuda_lbvh_builder` entry adds the native workspace/rebuild
  symbols and tests, while retaining its already frozen conditional Warp
  selection rule;
- the existing `temporary_warp_comparison_lane` entry adds the static stable-
  view interop binding and deletes it at terminal cleanup unless CUDA LBVH is
  separately productionized for the Warp product;
- the existing exports/examples/tests entry lists the migrated LBVH example and
  every C3 package export/call site that must collapse.

If CUDA is selected:

- retain the CPU-safe scene contract and Torch CUDA scene/LBVH data plane;
- make the native path the sole product owner after C4/B3 and product gates;
- delete Warp scene/BVH buffers, prepared Warp commands, Warp optical
  executors, static comparison bindings, and compatibility bridges;
- delete `build_cuda_lbvh_from_snapshot(...)` and the old Warp-snapshot LBVH
  adapters after their callers migrate to the native scene/LBVH path;
- keep no production import from `examples/`.

If Warp is selected:

- retain the CPU-safe scene contract, Warp scene/refit data plane, and active
  Warp regressions;
- delete Torch CUDA scene update, CUDA renderer/timeline, CUDA-only C3 tests,
  selectors, and comparison adapters;
- retain CUDA LBVH only if it independently passes the already frozen 20%
  build/refit and 10% dynamic-transaction benefit/maintenance gate, then move
  it to a production-owned name and keep long-term GPU tests;
- otherwise delete the CUDA LBVH module/source, static interop binding, and
  extension cache identity with the losing CUDA optical lane.

In either branch, remove paired-selection coordination and leave one thin
selected-backend durable regression lane. The implementation-neutral host
scene contract remains because it contains no backend selector, runtime buffer,
stream, event, or fallback.

## Acceptance checklist

C3 is accepted only when all of the following are true:

- one CPU-safe packed scene is the only scene-array materialization source and
  drives the a2 manifest plus independent Warp/Torch uploads;
- every result-capable manifest, readiness, runtime scene, stage,
  acceleration, binding, provenance digest, and receipt agrees on the same
  `scene_contract_sha256` before enqueue;
- replay-backed samples independently agree on `transform_replay_sha256` and
  `transform_frame_index`, while S rejects replay provenance;
- `dataflow_binding_sha256` proves the exact scene-update, acceleration, and
  renderer storage chain and rejects same-shaped substitute buffers.
- provenance hashes form the frozen source/storage -> stage evidence -> native
  execution binding -> receipt DAG, with no reverse aggregate reference;
- R0 uses `CanonicalFixedRenderFixtureHost` through a2 upload/readiness and
  exact fixture-to-render allocation binding, never a separate fixture upload;
- C3 factories prove one-shot sealing only; C4 alone creates final frame/slot
  bundles, validates lease generation, and supplies result-capable binding
  evidence.

1. one CPU-safe scene contract drives independent Warp and Torch uploads;
2. CUDA product scene/LBVH inputs and outputs are Torch-owned and never wrapped
   from Warp;
3. timed scene/refit/rebuild entries allocate, upload, materialize, bridge,
   load, query, wait, or synchronize nothing;
4. all CUDA extension outputs and workspaces are caller-owned and use fixed
   Tensor ABIs with indexed device/current-stream guards;
5. Warp measured commands are prepared with `record_cmd=True` and launch only
   on the frozen Warp stream;
6. dynamic Warp uses immutable CPU leaf-4 topology plus GPU refit, while CUDA
   uses the explicitly named full Morton/Karras rebuild;
7. static Warp/CUDA-LBVH interop creates views only during setup and submits on
   one proven native stream;
8. CUDA scene → LBVH → C2 render has no intermediate ready event and exactly
   one terminal event at transaction finish;
9. direct scene, topology, bounds, traversal, shade, empty/single-primitive,
   multi-device, ambient-device restoration, mutation, failure, and stale-slot
   tests pass;
10. lane-produced evidence reflects actual bound commands/storage/workspace and
    can derive the frozen `AccelerationReceipt` without copying gate values;
11. runtime hooks and source ABI checks cover every new real entry point;
12. old compatibility paths remain visibly ineligible and cannot be selected
    by an a3b lane;
13. cleanup-ledger entries make both terminal backend branches mechanically
    complete;
14. no a3b result, baseline, threshold change, backend decision, or product
    default is created or inspected.

## Review request

The P1/P2 re-review should first confirm that:

- the provenance hashes form a one-way canonical DAG and no stage evidence
  references aggregate native execution binding identity;
- `PackedOpticalSceneHost` is mechanically bound through a2 readiness, exact
  native allocations, C3 dataflow, and final receipts;
- R0 has an equally strict packed-scene -> fixed-render-fixture -> a2 upload ->
  renderer path with same-shape wrong-storage tests;
- replay hash/frame identity remains independent from scene, R0 fixture, and S
  publication provenance;
- C3 owns only factories/one-shot sealing proof, while C4 creates every final
  frame/slot bundle and validates lease generation without rebinding.

Please review whether:

1. the common host packer is a logical-data contract rather than a runtime
   bridge or permanent dual-backend abstraction;
2. the Warp refit and CUDA rebuild transactions exactly match the frozen gate
   identities and fair native-ready timing boundary;
3. prebound Warp commands and fixed CUDA ABIs actually remove all known timed
   allocation/materialization/synchronization points;
4. the CUDA scene/LBVH stages compose with C2 without an intermediate event or
   hidden completion;
5. the workspace query, ownership, validation, and zero/one-primitive behavior
   close the current CUDA LBVH allocation bugs;
6. static Warp/CUDA-LBVH interop is narrow, setup-only, same-stream, measurable,
   and removable under either terminal decision;
7. evidence and correctness tests prevent a lane from relabeling the wrong
   scene update, topology, refit/rebuild, storage, or stream as a valid row;
8. S remains honestly blocked on reviewed per-borrow composition rather than
   being simulated with replay or cached physics-slot views;
9. the C3a-e slices are bounded and leave C4, B3, a3b, shipping activation,
   performance inspection, and backend selection outside this proposal;
10. the cleanup contract leaves a clean single-backend repository whichever
    candidate is ultimately retained.
