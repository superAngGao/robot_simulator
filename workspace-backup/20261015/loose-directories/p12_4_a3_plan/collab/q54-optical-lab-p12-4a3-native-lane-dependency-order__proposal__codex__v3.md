# Q54 Optical Lab P12.4a3 Native-Lane Dependency Order Correction v3

Author: Codex

Date: 2026-08-12

Status: proposed correction for owner and Claude review

Supersedes: `q54-optical-lab-p12-4a3-native-lane-dependency-order__proposal__codex__v2.md`

Parent design: P12.4 proposal v5 (`46a5d33`)

Parent implementation: P12.4a2 (`1824fd90c1c84fa5633800cd5ff12a8e9b5014c5`)

## Owner summary

Retain the v1 correction: do not wrap the currently allocating, synchronizing,
and bridging product calls and call the result a valid R0-R2 benchmark. Split
a3 into eligibility/audit (`a3a`) and completed lanes (`a3b`), execute the
already-approved C0 and bounded C2/C3/C4/B3 prerequisites between them, and
keep a4 as the first review allowed to inspect a common-subset baseline.

v3 retains the v2 dynamic acceleration decision and separates two input and
timing contracts that must never be substituted for one another:

- native R1/R2 dynamic replay starts from an already-resident replay transform;
- shipping S starts immediately before a real `GpuEngine.step(...)` and crosses
  the reviewed physics publication boundary.

The Warp native replay transaction is:

```text
setup outside timing:
  CPU median/longest-axis leaf-4 topology in canonical source order
  -> one Warp-device topology upload

every measured R1/R2 dynamic replay transaction:
  preselected ready Warp transform frame
  -> preallocated Warp world-scene/AABB update
  -> Warp GPU BVH refit of the immutable topology
  -> render / delivery / reclaim endpoint required by the row
```

The CUDA native replay transaction is initially:

```text
setup outside timing:
  static local scene, replay transforms, and all scene/LBVH/CUB arenas

every measured R1/R2 dynamic replay transaction:
  preselected ready Torch transform frame
  -> preallocated Torch CUDA world-scene/AABB update
  -> full native CUDA Morton/Karras LBVH rebuild into leased output buffers
  -> render / delivery / reclaim endpoint required by the row
```

These are deliberately different production acceleration strategies. They are
not assigned the same acceleration identifier. R0 fixes topology/order and
isolates renderer kernels; R1 measures the complete proposed product
transaction, including the benefit or cost of its selected acceleration
strategy.

The corresponding shipping transaction is separate for each candidate:

```text
start shipping timer
-> real GpuEngine.step(...)
-> physics publish ring writes a physics-owned transform slot/event
-> bound optical consumer borrows that exact frame/slot
-> consumer stream waits for the publish event
-> zero-copy candidate-native views are bound to that borrowed slot
-> candidate-native scene update reads the physics-owned transforms
-> publication-owned consumer completion is recorded after the last reader
-> the same Warp-refit or CUDA-rebuild acceleration operation defined below
-> render / payload delivery / output reclaim
-> physics slot logical reclaim and required producer-stream reuse dependency
-> final shipping endpoint required by the row
```

No transform for the measured shipping frame may come from the replay buffers.
Only the scene-update and acceleration implementations are shared with R1; the
input provenance, timing start, publication wait, and slot lifecycle are not.

## Evidence from the current product path

`OpticalLabRenderWorkspace.prepare_dynamic_frame(...)` first constructs a base
BVH during session setup. When `base_bvh.stats.supports_refit` is true, every
dynamic frame invokes `refit_device_bvh_from_snapshot(...)`; it invokes the CPU
or CUDA rebuild path only when the chosen base topology does not support refit.

`build_device_bvh_from_snapshot(...)` produces a median-split topology with
`leaf_size=4` by default, stable source-order tie behavior, node depths, level
ranges, and `supports_refit=True`. `refit_device_bvh_from_snapshot(...)` then
updates leaf bounds and reduces internal bounds on Warp GPU kernels.

`build_cuda_lbvh_from_snapshot(...)` currently returns
`supports_refit=False`, so the CUDA-LBVH dynamic path rebuilds. Proposal v5 C3
also says to support dynamic rebuild first and add refit only when separately
implemented/measured. The first valid a3b candidate therefore uses rebuild,
not an unimplemented CUDA refit.

## Frozen acceleration vocabulary

`production_dynamic` and `production_physics` remain workload classes, not
acceleration implementation names. `production_dynamic` means native R1/R2
replay input. `production_physics` means real stepped-and-published S input;
the latter may never be satisfied by selecting a replay frame.

Before a3a produces any result-capable schema, replace the ambiguous lane
bindings:

```text
Warp production_dynamic:
  cpu_median_leaf4_topology_warp_refit_v1

Warp production_physics:
  cpu_median_leaf4_topology_warp_refit_v1

CUDA production_dynamic:
  cuda_morton_karras_lbvh_rebuild_v1

CUDA production_physics:
  cuda_morton_karras_lbvh_rebuild_v1
```

Do not retain `cpu_bvh` or generic `cuda_lbvh` as receipt identities for these
rows. Those names fail to say whether timed work is a build, rebuild, or refit.

The R0 identity remains `fixed_bvh_fixture`, and its fixed normalized topology
is independently allocated for each renderer before timing.

For the static Go2 R2 rows, both lanes retain the frozen `production_static`
workload class but must also use an exact acceleration identity. The intended
initial identity is:

```text
cuda_morton_karras_lbvh_rebuild_v1
```

for both renderer lanes, because the existing gate deliberately assigns the
CUDA LBVH acceleration axis to both. In the Warp-renderer lane, all AABB input
and BVH output views are created once outside timing, target the lane's
preallocated Warp-owned arenas, and enqueue the CUDA LBVH work on the same
underlying CUDA stream as Warp scene update/render. No per-frame
`wp.to_torch`, `wp.from_torch`, event bridge, or host/global synchronization is
allowed. The CUDA-renderer lane uses independently allocated Torch-owned
arenas. This shared acceleration algorithm is recorded separately from the
different renderer identities.

## Frozen scene-update and acceleration semantics

The following operations apply to the mandatory R1 dynamic rows. Once C0 and
the shipping product path are approved/implemented, S reuses the same
candidate-specific scene-update and acceleration operations only after it has
acquired the measured frame through physics publication.

| Field | Warp dynamic lane | CUDA dynamic lane |
|---|---|---|
| `topology_origin` | `cpu_median_longest_axis_leaf4_source_order_v1` | `cuda_morton30_karras_source_index_tiebreak_v1` |
| `topology_lifetime` | immutable for one prepared lane/session; rebuilt only between independent reset runs | one transaction/output-arena lease; contents overwritten only after reclaim |
| `timed_scene_update_operation` | `warp_local_scene_transform_and_aabb_out_v1` | `torch_cuda_local_scene_transform_and_aabb_out_v1` |
| `timed_acceleration_operation` | `warp_bvh_leaf_internal_refit_out_v1` | `cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1` |
| classification | `refit` | `rebuild` |
| timed host work | none | none |
| timed allocation | none | none, including CUB scratch |
| timed synchronization | native stream ordering and completion-event record only | native stream ordering and completion-event record only |
| timed bridge/view creation | none | none |

This is fair at the product-transaction level because:

- both lanes start with the same logical local geometry, transform frame,
  camera/light/profile request, and already-resident input readiness;
- both must update world geometry/AABBs for the measured frame;
- each pays the complete cost of the acceleration operation its proposed
  product path requires;
- neither receives world geometry, BVH bounds, or per-frame topology computed
  before timing;
- renderer-only performance remains separately comparable in R0 with identical
  normalized topology/order.

It would be less fair to force Warp to rebuild a topology its selected product
path legitimately refits, or to report CUDA rebuild as a refit. Algorithmic
quality and update cost are part of R1 by design.

## Separate input and timing contracts

| Field | R1/R2 dynamic replay | S physics-published shipping |
|---|---|---|
| `input_contract` | `canonical_native_replay_v1` | `gpu_engine_published_frame_v1` |
| measured transform owner | candidate-native benchmark replay arena | physics publish-ring slot |
| transform preparation | deterministic sequence generated and uploaded before timing | produced by the measured `GpuEngine.step(...)` only |
| timed start | immediately before candidate-native scene update for the selected ready replay frame | immediately before the real `GpuEngine.step(...)` call |
| handoff in timed region | none | publish, borrow, consumer-stream wait, and zero-copy view binding |
| terminal lifecycle | native/output lease endpoint required by R1/R2 | payload/output endpoint plus physics consumer completion, logical reclaim, and final producer reuse dependency required by S |
| allowed source provenance | canonical replay manifest/frame index | exact physics run/step/frame/slot identity |
| forbidden source provenance | physics publication substituted for a replay row | replay buffer, replay selector, cloned published frame, or cached prior physics frame |

The lane execution receipt and frozen row semantics must carry and match
`input_contract`. A record with the wrong input contract is invalid even if its
scene, acceleration, render, and output values are numerically correct.

## Work allowed before the native-ready boundary

Common to both lanes:

- canonical logical case construction and content hashing;
- independent native upload of local geometry, metadata, materials, lights,
  and cameras;
- extension/module compilation and all warmup frames;
- allocation of scene, AABB, acceleration, renderer, diagnostic, delivery,
  pinned-host, and runtime scratch arenas;
- creation and validation of stable same-allocation runtime views needed by the
  selected acceleration/delivery implementation;
- direct untimed input/output/topology correctness validation.

R1/R2 replay-only setup:

- generate and upload the entire deterministic transform replay sequence into
  each lane's independent native replay arena;
- select the next already-ready replay-frame view/index outside timing.

S shipping-only setup:

- initialize and reset the deterministic physics state, action sequence, seed,
  camera sequence, publish ring, and reviewed C0 consumer binding;
- allocate the optical arenas and establish the stable external-stream consumer
  binding; the measured slot's zero-copy transform views are created/bound only
  after borrow inside the timed shipping handoff;
- perform setup/warmup steps only in a separate reset run whose frames and
  slots cannot be used as measured S inputs.

S setup may prepare static metadata, but it may not create/bind the measured
sample's transform view, populate, clone, cache, upload, select, or pre-complete
the measured frame's transform contents.
Those contents must be produced by the measured physics step and made readable
by its publish event. No per-slot view cache may move C0's required zero-copy
binding work out of R-ship/A-ship.

Warp-only setup:

- construct the CPU median leaf-4 topology from the canonical setup frame;
- upload its immutable topology and level ranges once;
- allocate separate mutable node-bounds buffers for refit.

CUDA-only setup:

- query and allocate maximum CUB scratch for the frozen primitive count;
- allocate both LBVH buffer slots needed by the output lease policy;
- initialize constants that do not depend on the measured transform frame.

Static Go2 shared-CUDA-acceleration setup:

- create stable Torch views of Warp-owned AABB/LBVH arenas for the Warp
  renderer lane and stable Warp views of its LBVH output arena;
- bind both runtimes to one underlying CUDA stream for that lane;
- create the CUDA-renderer lane's independent Torch-owned arenas separately.

## Work forbidden before timing for a measured frame

- transforming that frame's local geometry to world geometry;
- calculating that frame's primitive AABBs;
- refitting node bounds for that frame;
- computing scene bounds, Morton keys/order, topology, or node bounds for that
  frame's CUDA rebuild;
- rendering, packing, D2H submission, host readiness, or reclaim work belonging
  to that frame;
- pre-recording the sample's completion event;
- reusing a scene/BVH/output slot that has not reached its required reclaim
  endpoint.

Additionally forbidden for S:

- reading a measured transform from `canonical_native_replay_v1` or any replay,
  cache, clone, or previously published frame;
- calling physics step/publish, borrowing the physics slot, or enqueueing its
  consumer-stream wait or binding its zero-copy transform views before the
  shipping timer starts;
- ending the measurement at host image readiness while physics consumer
  completion, logical reclaim, final payload drain, or final reclaim required
  by the row is still outstanding.

## Required acceleration evidence in the lane receipt

Add a versioned immutable `AccelerationReceipt` produced by the lane, not the
coordinator. It must contain at least:

```text
acceleration_schema_version
acceleration_id
workload_class
topology_origin
topology_lifetime
topology_validation_sha256
timed_scene_update_operation
timed_acceleration_operation
operation_classification        # none / refit / rebuild
scene_update_count
refit_count
rebuild_count
primitive_count
node_count
leaf_count
scene_update_kernel_launch_count
acceleration_kernel_launch_count
device_allocation_count
host_materialization_count
host_upload_bytes
device_to_host_bytes
global_synchronize_count
bridge_creation_count
completion_event_recorded
instrumentation_trace_sha256
```

The enclosing lane `ExecutionReceipt` also records `input_contract` and
`input_provenance_sha256`. This separates input identity from acceleration
identity while cryptographically binding both receipts to the same sample.

Row-specific invariants:

- R0: one render transaction; classification `none`; refit/rebuild counts zero;
- Warp R1 dynamic: scene update 1, refit 1, rebuild 0;
- CUDA R1 dynamic: scene update 1, refit 0, rebuild 1;
- every R2 row: exactly one fresh scene-update/acceleration/render transaction
  before its delivery operations;
- all successful primary rows: allocations, host materialization/upload/D2H
  inside native compute, global synchronizations, and bridge creation are zero;
- expected D2H occurs only in the delivery portion of host-payload R2 and is
  recorded as payload bytes, not hidden in acceleration evidence.

The topology validation hash comes from a direct, untimed validation pass over
the exact case/frame/topology contract. Timed samples bind that validation
artifact and record actual operation counters; they do not stage a topology to
the host merely to calculate a per-sample hash.

### Additional shipping publication evidence

Acceleration evidence cannot prove an S product path. Each S sample therefore
also requires a lane-produced, immutable `ShippingPublicationReceipt` with at
least:

```text
publication_schema_version
input_contract                    # gpu_engine_published_frame_v1
physics_run_id
physics_step_index
physics_frame_id
physics_slot_id
physics_device_ordinal
physics_step_count
publish_event_token
publish_event_recorded
producer_publish_completion_recorded
consumer_binding_id
consumer_stream_token
borrow_count
consumer_stream_wait_enqueued
zero_copy_transform_binding
last_physics_buffer_reader_operation
consumer_done_event_token
consumer_completion_recorded
slot_logically_reclaimable
producer_reuse_wait_enqueued_when_reused
done_event_retained_until_reuse_dependency
physical_reclaim_evidence          # producer wait before reuse / retained
final_payload_drain_complete
final_reclaim_complete
replay_source_count
host_poll_count
host_synchronize_count
publication_trace_sha256
```

The event/stream tokens are run-local identities used to prove ordering; they
are not treated as stable values across processes. The receipt is valid only
when:

- `physics_step_count == 1`, `borrow_count == 1`, and
  `replay_source_count == 0` for the measured sample;
- frame/slot identity matches the frame returned by that measured physics step;
- producer publication completion is represented by that frame's recorded
  publish event, and the bound consumer stream waited on that exact event;
- the done event was recorded after the last enqueued reader of physics-owned
  transforms and registered through the public C0 contract;
- logical reclaim is reported separately from physical safety. Physical
  reclaim evidence is the producer-stream wait enqueued before overwrite, or
  retention of the done event and slot when no reuse occurs; it is not a claim
  that a host query observed GPU event completion;
- before an overwritten slot is reused, the producer stream has enqueued its
  device-side wait and the done event remained alive until then;
- all row-required final drain/reclaim flags are true without host polling or a
  host/global synchronize added for publication.

The coordinator may add run/repeat/sample timing metadata but may not populate
or rewrite execution, acceleration, publication, payload, or lifecycle facts.

## Corrected implementation order

### P12.4a3a — eligibility and concrete entry-point audit

Deliver only:

- exact lane/acceleration identities above;
- versioned eligibility, input-contract, acceleration-receipt, and shipping-
  publication-receipt schemas;
- runtime instrumentation coverage for allocation, upload/materialization,
  synchronization, bridge/view creation, extension load, stream wait,
  scene-update, acceleration, render, pack, delivery, and reclaim entries;
- preflight that rejects an ineligible row before timed submission and cannot
  emit a schema-valid result record;
- exact Go2 asset-manifest preflight; absent/mismatched asset is a hard failure,
  never a surrogate row;
- negative tests against current invalid product calls;
- cleanup-ledger entries for both temporary lane adapters.

a3a does not write baseline samples.

### P12.4c0a/c0b — public CUDA publication prerequisite

Bounded deliverables:

- reviewed public external-CUDA-stream consumer binding;
- publication-owned publish/done event lifetime;
- ordered borrow/complete and slot reclaim;
- lane-visible publication instrumentation sufficient to produce the frozen
  S receipt without private ring access or host observation;
- ambient-device restoration and device/stream validation;
- ordinary-exception completion versus poisoned CUDA/event-record failure;
- close/remove/final-drain lifecycle tests;
- no host polling/synchronize, private ring mutation, or Torch event in ring.

This slice does not enable the CUDA optical product path.

### Required P12.4c2 portion — native execution timeline

Bounded deliverables:

- explicit indexed Torch stream for scene/LBVH/render/pack;
- no target-wide synchronization in internal native calls;
- native completion events returned without host wait;
- current-device guard/restoration;
- out-style extension entry points launch on the selected current stream.

No profile optimization or kernel fusion is included.

### Required P12.4c3 portion — native scene and acceleration

Bounded deliverables:

- Torch-owned CUDA static/local scene and per-frame transform inputs;
- preallocated Torch CUDA world-scene/AABB update;
- out-style CUDA Morton/Karras LBVH rebuild with caller-owned output and CUB
  scratch buffers;
- preallocated Warp world-scene/AABB update and GPU refit path for the exact
  CPU-origin topology contract;
- stable same-stream, prebound views required for the Warp-renderer + CUDA-LBVH
  static R2 lane;
- direct topology/bounds/traversal parity outside timing.

No CUDA refit, alternative builder, SAH tuning, or production physics enablement
is included.

### Required P12.4c4 portion — benchmarked arenas and leases

Bounded deliverables:

- reusable ray, geometry, shade, diagnostics, scene, AABB, LBVH, RGB8, and
  delivery buffers for only the frozen R0-R2 shapes/profiles;
- runtime scratch and CUB workspace preallocation;
- output-slot lease/reclaim with reuse blocked until native/delivery consumers
  finish;
- resize, device/profile change, exception, double-release, active-close, and
  final-drain tests;
- no general product backend registry.

No arbitrary-resolution arena policy beyond correctness for the frozen cases
is included.

### Required P12.4b3 portion — frozen R2 delivery matrix

Bounded deliverables:

- out-style CUDA RGB8 pack;
- preallocated synchronous host staging for RGB/full/RGB8;
- preallocated ordered Torch async RGB/RGB8 rings at frozen depth;
- final payload drain and final output/staging reclaim;
- equivalent payload bytes, ordering, backpressure policy, and host consumer
  work across matched lanes;
- prebound Warp-result Torch views outside timing where the retained Warp
  delivery lane requires them.

No full async, image/video encoding, disk write, or product-default change is
included.

### P12.4a3b — completed real R0-R2 lanes

Only after all prerequisite tests pass:

- materialize exact canonical cases independently in Warp/Torch ownership;
- certify native readiness after compile, arenas, views, validation, and 10
  warmup frames;
- execute R0, dynamic replay R1, and every frozen native R2 payload/policy row;
- produce lane-owned execution, acceleration, payload, and lifecycle receipts;
- prove direct CPU/Warp/CUDA parity outside primary measurements;
- prove every successful timed trace has zero forbidden contaminating entries;
- keep primary results unavailable to replacement analysis until a4 approval.

a3b does not execute or label a replay-backed sample as S. Its dynamic input
contract is `canonical_native_replay_v1`.

### P12.4a4 — validity review and baseline authorization

a4 reviews boundaries, exact assets, instrumentation coverage, acceleration
identity, R2 fresh-transaction traces, async drain/reclaim, and parity. Only an
approved a4 may authorize inspection of the common-subset baseline.

### Later P12.4b4/d — real shipping S activation and decision

S becomes executable only after the public C0 contract and the selected
candidate's native scene/acceleration/render/delivery path pass their lifecycle
tests. Each paired lane runs an independent reset physics sequence and starts
timing immediately before real `GpuEngine.step(...)`. A shipping result is
schema-invalid unless its `input_contract` is
`gpu_engine_published_frame_v1`, its `ShippingPublicationReceipt` satisfies all
invariants above, and its terminal endpoint includes the row-required payload
drain and final reclaim. Replay-backed samples remain useful R1/R2 diagnostics
but can never enter R-ship or A-ship.

## Later acceleration changes

The initial CUDA dynamic identity is rebuild. Replacing it with CUDA refit is
not a silent implementation detail: it changes topology origin/lifetime and
timed work. It therefore requires:

- a new exact acceleration identifier and row-semantics hash;
- a predeclared optimization slice within the existing continuation budget;
- correctness/topology/lifetime validation before its performance rerun;
- retention of the original raw artifact and explicit candidate-version
  comparison.

It may not rewrite an already observed baseline under the old identifier.

## Unchanged rejections and cleanup

Still forbidden:

- benchmark-only copies of product kernels;
- hiding C++ allocation because the Python guard cannot see it;
- relaxing the frozen timed boundary after observing results;
- surrogate Go2 geometry under a canonical Go2 row;
- product imports of `benchmarks.optical`;
- a permanent paired selector or dual-backend fallback.

Whichever backend wins, P12.4e deletes the losing lane and comparison-only
coordination. The selected exact acceleration/renderer identities remain in the
durable regression adapter. CUDA-LBVH retention with a Warp-selected product
still must pass its independent quantitative and productionization gate.

## Acceptance requested

Please approve or request changes on the following before a3a implementation:

1. Warp dynamic/physics uses CPU-origin immutable topology plus timed Warp GPU
   refit and receives the exact new acceleration identity;
2. CUDA dynamic/physics initially uses a timed native CUDA LBVH rebuild and a
   different exact identity;
3. the different operations are intentionally compared only at complete R1/S
   product-transaction level, while R0 fixes topology for renderer comparison;
4. the allowed setup work, forbidden per-frame precomputation, and required
   acceleration receipt evidence are frozen as above;
5. R1/R2 replay and S physics publication use different frozen input contracts,
   timing starts, and provenance; S requires independent publication/lifecycle
   evidence and rejects all replay sources;
6. the C0/C2/C3/C4/B3 prerequisites are limited to the explicit deliverables
   listed above;
7. a3a cannot emit baseline records, and a4 remains the first native baseline
   authorization point.
