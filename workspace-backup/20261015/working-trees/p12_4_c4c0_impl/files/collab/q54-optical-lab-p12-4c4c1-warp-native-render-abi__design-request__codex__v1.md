# Q54 Optical Lab P12.4 C4c1 Warp Native Render ABI

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c1 design request
- Author: Codex
- Version: v1
- Date: 2026-09-20
- Base: `591ffc57fd160e681d53bca3e8642cf296c49605`
- Status: design request for Claude review

This document narrows the approved C4c v7 plan to C4c1 only. C4c0 is approved
at `591ffc57`. C4c1 establishes the Warp render ABI and proves its correctness;
it does not activate the closed worker's GPU preparation or create an arena,
transaction, completion ticket, benchmark sample, or result capability.

## 1. Outcome

C4c1 will leave the repository with one Warp implementation of pinhole ray
generation, BVH first-hit, shadow traversal, and direct-light shading. The
existing compatibility executors and the future Warp arena command path will
use that same implementation.

The slice delivers:

- one production source for the reusable Warp kernels and device functions;
- an exact caller-owned `WarpDirectLightBuffers` ABI;
- deterministic per-submission initialization for every output and diagnostic
  counter;
- four fixed shade entry ABIs selected before command recording;
- a prepared immutable factory and a frame/slot-specific recorded command
  bundle;
- independent, canonical render-stage evidence tied to the actual storage and
  dispatch plan;
- a blocking, benchmark-ineligible compatibility/validation driver; and
- direct ray, output, profile, sentinel-reuse, and oracle-parity tests.

C4c1 does not deliver:

- a Warp transaction template or terminal event;
- C4 slot/generation/lease authority;
- diagnostic timing events;
- scene-update/refit composition inside a transaction;
- the static CUDA-LBVH R2 binding;
- host payload, readback, external holds, or B3;
- a worker IPC operation that can prepare or execute GPU work; or
- a3b/result/baseline access.

Those boundaries are structural. A C4c1 command bundle can be exercised only
by the explicit blocking compatibility/validation path in ordinary GPU tests.
It cannot produce the completion and lifecycle evidence required by the native
runner.

## 2. Frozen parent contracts

C4c1 remains subordinate to:

- backend parity/performance gate v5;
- native-lane dependency order v3;
- a2 canonical native input/readiness contracts;
- a3a runtime instrumentation;
- C3a provenance DAG and C3b/C3e command/storage contracts;
- C4 arena design plus the external-hold addendum;
- approved C4a/C4b implementations;
- approved C4c design v7; and
- approved C4c0 closed-worker/setup lifecycle at `591ffc57`.

No workload, threshold, profile, acceleration identity, or payload contract is
changed by this slice.

The following gates remain unchanged and false:

```text
closed_worker_runtime_contract.gpu_preparation_enabled = false
native_result_activation.result_capable                = false
a3b result outlet                                       = closed
baseline read/write outlet                              = closed
host-payload arena seal                                 = closed
```

`PREPARE_FROZEN_ROW` continues to return
`GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW`. C4c1 does not add its new
module to the closed-worker source allowlist and does not change the worker
protocol, manifest version, request dispatcher, or activation fields. C4c2
must make any worker-source expansion and GPU-preparation activation a separate
reviewed decision.

## 3. Current code facts

`optics/warp_execution.py` currently contains both public executors and their
kernel implementation. The direct-light camera path:

- converts camera matrices through NumPy on every call;
- allocates ray, geometry, shade, and diagnostic arrays on every call;
- initializes FULL geometry outputs with host NumPy arrays;
- initializes reduced profiles with only a hit-mask kernel;
- uses one shade kernel with runtime shadow and traversal-stat flags;
- aliases all nine disabled traversal counters to one primary scalar;
- creates readiness/profiling events during execution; and
- does not expose a reusable recorded-command ABI.

The mathematics and output conventions are already the Warp optical oracle.
C4c1 changes ownership, initialization, and dispatch shape; it does not change
hit tie-breaking, intersection equations, lighting equations, shadow bias,
attenuation, luminance, or public channel semantics.

C3 already demonstrates the required recorded-command pattern with
`record_cmd=True`. C4b demonstrates caller-owned output validation and the
one-way provenance DAG. C4c1 adopts those patterns without importing CUDA's
Torch transaction classes or creating a generic backend registry.

## 4. Module ownership and single source

A new production module, `optics/warp_native_render.py`, owns:

- all Warp device functions and kernels used by BVH direct-light rendering;
- the kernel ABI/version registry;
- camera scalar normalization used by recorded commands;
- render input/output storage validation;
- the caller-owned buffer type;
- render evidence, factory, and command bundle; and
- the compatibility/validation command driver.

`optics/warp_execution.py` remains the public compatibility-executor module.
It imports the shared kernel objects and native ABI helpers; it must not keep a
second definition of pinhole raygen, initialized BVH first-hit, shadow
traversal, or direct-light shade. Brute-force and unrelated legacy kernels may
remain where they are.

The migration is mechanical:

1. move the reusable device functions and kernel definitions without changing
   their equations;
2. update compatibility executors to call/import those exact kernel objects;
3. add the caller-owned recorded-command entry style; and
4. delete the superseded direct-light kernel definitions from
   `warp_execution.py` in the same slice.

A source-ownership test scans the production modules and requires exactly one
definition of every frozen render entry. Importing the new module remains safe
on a CPU-only host: absence of Warp defers the runtime error until preparation
or construction, matching current optional-dependency behavior.

There is no benchmark-to-production dependency. Production code does not
import `benchmarks.optical`, its package initializer, the runner, gate JSON, or
test helpers.

## 5. Exact caller-owned buffer ABI

`WarpDirectLightBuffers` is an immutable typed owner/view over one physical
render storage graph. The ordered ABI is:

| Group | Name | Warp dtype | Shape |
|---|---|---|---|
| rays | `origins` | `wp.float32` | `[num_rays, 3]` |
| rays | `directions` | `wp.float32` | `[num_rays, 3]` |
| geometry | `hit_mask` | `wp.int32` | `[num_rays]` |
| geometry | `range_m` | `wp.float32` | `[num_rays]` |
| geometry | `position_world` | `wp.float32` | `[num_rays, 3]` |
| geometry | `normal_world` | `wp.float32` | `[num_rays, 3]` |
| geometry | `numeric_instance_id` | `wp.int32` | `[num_rays]` |
| geometry | `material_index` | `wp.int32` | `[num_rays]` |
| BVH primary | `bvh_stack_overflow_count` | `wp.int32` | `[1]` |
| BVH primary | `bvh_max_stack_depth` | `wp.int32` | `[1]` |
| shade | `rgb` | `wp.float32` | `[num_rays, 3]` |
| shade | `intensity` | `wp.float32` | `[num_rays]` |
| shadow primary | `shadow_stack_overflow_count` | `wp.int32` | `[1]` |
| shadow primary | `shadow_max_stack_depth` | `wp.int32` | `[1]` |

When `traversal_diagnostics=True`, the buffer additionally owns exactly nine
distinct `[1]`, `wp.int32` arrays in
`SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES` order. When false, it owns none;
there are no dummy arrays, aliases, fallback containers, or optional positions
in the selected primary shade ABI.

Every physical buffer above is present for all three direct-light profiles.
Profile selection controls only which completed channels a compatibility or
future generation-scoped result view exposes. It never changes allocation or
the physical render graph. `direct_light_full`, `rgb_preview`, and
`render_only` therefore execute identical raygen/first-hit/shade work for the
same shadow/diagnostic mode.

Full validation occurs before command recording and freezes:

- exact indexed Warp CUDA device;
- exact object identity, base pointer, active byte interval, dtype, rank,
  shape, contiguity, and logical semantic for every array;
- exact ray count and scalar counts;
- pairwise-disjoint output intervals;
- no overlap between mutable outputs and any render input; and
- the complete strong-owner graph.

For a camera binding, `origins` and `directions` are mutable raygen outputs.
For the compatibility-only `device_rays` binding, they are exact read-only
inputs and must already be resident on the target device. Both roles enter the
storage manifest explicitly; neither may overlap geometry or shade outputs.

Different outputs may share one arena allocation only when their recorded
active byte intervals are disjoint and the manifest retains the same physical
owner; overlap remains forbidden. The only cross-runtime aliases permitted in
later C4c3 are the already reviewed setup-time Warp/Torch views of static LBVH
storage. Those are input views, are recorded explicitly, and do not relax
output disjointness.

An allocation helper may exist for compatibility and tests only. It executes
outside any native enqueue and allocates with `wp.empty`; deterministic values
come from recorded device commands, not allocator freshness or host-created
NumPy arrays. C4c2 supplies final arena-owned buffers instead of using the
helper.

## 6. Deterministic initialization contract

Correctness after reuse must not depend on a new allocation. The frozen camera
command order is:

```text
pinhole raygen
-> reset two BVH primary counters
-> initialized BVH first-hit
-> reset two shadow primary counters [plus nine overlay counters when enabled]
-> fixed-ABI direct-light shade
```

Raygen overwrites every origin and direction in row-major
`py * width + px` order.

The initialized first-hit kernel writes miss defaults for all six geometry
outputs before testing any primitive:

```text
hit_mask            = 0
range_m              = +inf
position_world       = NaN
normal_world         = NaN
numeric_instance_id  = 0
material_index       = 0
```

Hits then overwrite all six values using the existing tie-breaking rules. The
kernel does not rely on a preceding host upload, `wp.zero_`, fresh `wp.empty`,
or a profile-specific partial initializer.

Two fixed one-thread reset kernels are allowed:

- BVH primary reset: exactly two scalar arrays;
- shadow reset: either exactly two primary arrays or exactly two primary plus
  nine distinct traversal arrays.

The reset variant is selected while sealing. No Python list/dict construction,
dynamic argument selection, or dummy storage occurs when recorded commands are
launched.

Shade overwrites every RGB and intensity element, including background misses.
No-shadow primary and overlay counters remain the explicit zeros written by
their reset command.

Zero-ray low-level `device_rays` bindings remain legal. They omit every per-ray launch
but still execute both scalar reset commands. Evidence records that exact
two-command plan. Product camera specifications remain subject to their current
positive-dimension validation; the zero-ray path exists to make the ABI and
reusable storage contract total, not to add a new workload row.

## 7. Four fixed shade ABIs

Seal selects exactly one of these entry identities:

```text
warp_direct_light_no_shadow_primary_v1
warp_direct_light_no_shadow_diagnostics_v1
warp_direct_light_with_shadows_primary_v1
warp_direct_light_with_shadows_diagnostics_v1
```

The rules are:

- all four receive the two distinct primary shadow counters;
- only the two diagnostic variants receive the nine traversal counters, in
  canonical fixed positions;
- no-shadow variants have no shadow-scene/BVH argument positions;
- shadow variants receive the exact world geometry, role masks, BVH topology/
  bounds, role mask, and shadow-bias inputs;
- selection is complete before `wp.launch(..., record_cmd=True)`;
- the recorded command contains no runtime shadow or diagnostics mode flag;
  and
- the launch path never builds a dynamic array collection.

The four kernels are ABI adapters over shared Warp device functions. Light
vector normalization, attenuation, BRDF/color accumulation, luminance,
intersection, tie-breaking, BVH traversal, and shadow visibility each retain a
single production definition. Compatibility and recorded-command entry styles
use the same four kernel objects; variant count does not fork the mathematics.

The primary and diagnostic reset commands are also separate fixed entries.
Diagnostics-off code contains no reference to the nine overlay arrays.

## 8. Camera and render input binding

`WarpCameraParameters` is frozen outside execution from
`OpticalPinholeCameraSpec`. It contains only normalized immutable scalars:

- width, height, fx, fy, cx, cy;
- float32 camera origin and row-major 3x3 rotation;
- finite maximum distance;
- sensor role and the role mask derived from the exact scene role table; and
- frame, environment, sensor, and camera-contract identity used in evidence.

NumPy conversion, finiteness checks, role lookup, and digest construction occur
before command recording. The command path receives only fixed scalar values.

The factory binds these exact runtime owners:

- one `WarpPreparedNativeInput` plus its registered readiness authority;
- the exact `DeviceOpticalScene` built over that prepared input's scene arrays;
- one explicit indexed Warp CUDA device and strong non-default stream owner;
- one exact `WarpWorldSceneBuffers` instance;
- one exact `DeviceOpticalBvh` instance;
- one camera binding; and
- one exact `WarpDirectLightBuffers` instance.

C4c1 adds the Warp-symmetric prepared/readiness authority check to
`optical_worker.native_contract`. It accepts only the exact registered prepared
object, exact canonical source/manifest identity, exact registered readiness
object, and matching backend/device/content/topology/scene/replay-or-fixture
facts. Class names, duck typing, copied dataclasses, caller-provided hashes, and
same-content reuploads do not satisfy it.

The local scene arrays must be the prepared arrays themselves or an approved
zero-copy Warp view whose base allocation and active interval match the
prepared manifest. No upload, copy, DLPack conversion, or Warp/Torch view is
created by renderer preparation or sealing.

World geometry and BVH arrays are actual command inputs and their exact storage
manifests enter render evidence. In C4c1 they may be a canonical R0 fixed
fixture or reviewed C3-produced R1 storage. C4c1 does not claim that render
evidence alone proves the full scene-update/acceleration dataflow. C4c2 must
bind the render bundle to exact C3 stage objects, ordered stage evidence, C4
slot/generation, and live lease before any supported measured submission.
C4c3 separately adds the static CUDA-LBVH R2 ownership proof.

This preserves the one-way provenance DAG:

```text
a2 prepared/readiness facts
        + actual scene/world/BVH/render storage
        -> independent render-stage evidence
        -> C4c2 transaction/arena binding
        -> later lane receipt
```

Render-stage evidence never references its own hash, a future transaction,
arena, lease, completion, or outer execution binding.

## 9. Factory, recorded commands, and sealing

The conceptual API is:

```python
factory = WarpDirectLightRenderCommandFactory.prepare(
    prepared_input=prepared,
    readiness=readiness,
    local_scene=scene,
    device=device,
    stream=stream,
)

bundle = factory.seal_camera(
    world=world,
    bvh=bvh,
    camera=camera_parameters,
    buffers=buffers,
    output_profile=profile,
    shadows=shadows,
    traversal_diagnostics=traversal_diagnostics,
    frame_provenance=frame_provenance,
)
```

Exact spelling may change, but the split is fixed.

A second `seal_device_rays(...)` entry preserves the existing executor's
explicit-ray compatibility and supplies the zero-ray ABI test. It binds exact
resident `origins`/`directions`, records no raygen command, and is permanently
benchmark-ineligible. All C4c2 R0/R1 product candidates use `seal_camera()`.

`prepare()` runs outside timing and performs:

- exact a2/readiness/local-scene/device/stream validation;
- Warp initialization, kernel load/compile, and warmup;
- capture of the kernel ABI identity and trusted source digest; and
- a one-way immutable factory seal.

`seal_camera()` runs outside timing and performs:

- camera, world, BVH, output, overlap, and provenance validation;
- selection of reset and shade entry variants;
- creation of an ordered immutable command descriptor tuple;
- `wp.launch(..., record_tape=False, record_cmd=True)` for each descriptor;
- construction of independent evidence from the actual bound objects; and
- a one-way immutable bundle seal.

Each recorded command has a frozen descriptor containing its semantic phase,
entry ID, kernel object identity, launch object identity, dimension, exact
ordered argument semantics, exact scalar digest, and exact array-object tuple.
The bundle separately freezes the strong owners and complete storage
signatures. The descriptor is validation metadata, not a second executable
representation.

`validate_bound()` reconstructs the descriptor plan, storage manifests,
stream/device facts, and evidence from the live bound graph. It rejects before
any launch if an entry, argument, array, scalar, stream, owner, command, order,
provenance value, or signature baseline drifted.

Both factory and bundle seals are one-way:

- ordinary assignment/deletion after seal is rejected;
- supported code cannot set `_sealed` back to false or replace the original
  signature;
- a second freeze is rejected; and
- `object.__setattr__()` drift of a live binding field, with the frozen
  baseline intact, is found by full validation.

The trusted-process boundary from C4c v7 still applies. Python object privacy
or a signature field is not described as a security sandbox against arbitrary
code already executing in the process. C4c1 is safe because the product worker
cannot load or execute this path yet and the native runner remains closed.
C4c2 must admit only the reviewed driver/stage modules into the closed worker
and connect supported execution to live C4 authority.

In particular, C4c1 does not claim resistance to hostile code that uses
`object.__setattr__()` to rewrite both a live object graph and every validation
baseline. Such code is excluded by the approved worker trust model. Negative
tests cover accidental/single-point drift and supported-API mutation, while
C4c2's source allowlist and fixed IPC exclude arbitrary mutation callbacks from
the product execution domain.

## 10. C4c1 execution surface

The command bundle exposes validation and evidence, but no completion ticket,
terminal event, result receipt, or arena authority.

An explicit `WarpDirectLightCompatibilityDriver` is the only supported C4c1
executor for a sealed bundle. It is deliberately benchmark-ineligible. It may:

- perform compatibility input-event waits before execution;
- launch the already recorded commands on their exact stream;
- create/record/synchronize compatibility profiling or readiness events; and
- construct an `OpticalComputeResult` whose channels follow the selected
  profile.

Those allocation/wait operations are outside the native command launch and
make the driver ineligible for a3b. It produces no native execution receipt.

The bundle's launch leaf itself performs only:

```text
fixed-count lightweight sealed/state/owner identity checks
-> recorded Launch.launch(stream=exact_stream), in frozen order
```

The compatibility driver calls full `validate_bound()` before entering this
leaf. The leaf does not rebuild a manifest, hash storage, walk prepared scene
arrays, or construct command/argument collections. Its checks are fixed by the
five-or-fewer command ABI and do not scale with rays, primitives, replay-frame
count, or arena depth. C4c2 will add lease/transaction identity checks without
moving full validation into measured acquire or enqueue.

It performs no allocation, upload, host materialization, interop conversion,
extension/module load, event construction/record/query, host synchronization,
device synchronization, profile filtering, or result construction. A launch
exception poisons the bundle, which cannot be launched again. C4c2 will route
the same failure into transaction and arena poison; C4c1 does not invent a
recoverable lifecycle.

Raw `wp.Launch`, Warp CDLL functions, and object internals remain reachable to
arbitrary same-process Python. As approved in C4c v7, isolation against such
code comes from the later frozen worker allowlist and non-object IPC, not from
underscores or wrappers. C4c1 neither weakens nor claims to complete that
boundary.

## 11. Evidence contract

`WarpDirectLightRenderCommandEvidence` is strict, immutable, canonically
serializable, and has a round-trip validator. It contains independent facts:

```text
schema_version
operation_id = warp_bvh_direct_light
kernel_abi_id and trusted kernel-source SHA-256
device_ordinal and native stream handle
native-input manifest and readiness SHA-256
scene contract and input contract
fixed-fixture OR replay hash/frame identity
camera contract SHA-256 and ray count
ray source = camera OR device_rays
output profile, shadows, traversal_diagnostics
scene/world/BVH input-storage SHA-256
render-output-storage SHA-256
ordered reset/raygen/first-hit/shade entry IDs
ordered command-plan SHA-256 and dispatch count
```

`camera` evidence requires the camera hash and the raygen entry.
`device_rays` evidence requires an exact ray-input storage hash, has no camera
hash or raygen entry, and cannot satisfy a later C4c2 row binding.

The source SHA is computed by the reviewed preparation path from the production
module file. It is never accepted as a caller-provided label. Later C4c2 also
requires it to match the closed-worker source manifest.

The fixed-fixture and replay branches are mutually exclusive and follow the
frozen input contract:

- `fixed_native_scene_v1`: fixed-render-fixture hash present; replay hash and
  frame absent;
- `canonical_native_replay_v1`: replay hash and frame present; fixed fixture
  absent.

Unknown fields, enums, malformed SHA-256 values, noncanonical encodings or
types, contradictory provenance branches, wrong dispatch counts, and
self/outer-binding references are rejected. The evidence SHA-256 is computed
outside the evidence payload and is referenced only by later outer evidence.

The bundle validator rebuilds evidence from its actual owners. A caller cannot
make a different same-shape buffer, scene, BVH, command, profile, or shade
variant valid merely by copying the expected hashes into a dataclass.

## 12. Profile and result-view rules

The compatibility result exposes exactly:

- `DIRECT_LIGHT_FULL`: all formally guaranteed per-ray and four primary scalar
  channels, including `material_index`;
- `RGB_PREVIEW`: `hit_mask`, `rgb`, and the four primary scalar channels; and
- `RENDER_ONLY`: the four primary scalar channels.

When traversal diagnostics are enabled, the nine formal diagnostic-overlay
scalars are appended after primary profile filtering. They are not part of the
primary payload contract and do not alter output-profile identity.

The buffer object itself is not a result view and does not authorize arbitrary
consumers to read hidden channels. C4c2 will expose generation-scoped device
views only after exact terminal resolution. C4c1 parity tests may inspect all
physical arrays after the blocking compatibility driver has synchronized.

Traversal-counter diagnostics and diagnostic timing events are different
features. C4c1 implements only traversal counters. R0 three-pair and R1/R2
five-pair timing events belong to C4c2's transaction and are not allocated,
recorded, or represented by this slice.

## 13. Compatibility guarantees

The existing public executors retain their signatures and supported profiles:

- `GpuDeviceBvhOpticalExecutor` continues to support geometry first-hit;
- `GpuDeviceBvhDirectLightOpticalExecutor.execute()` continues to accept
  uploaded explicit rays;
- `GpuDeviceBvhDirectLightOpticalExecutor.execute_camera()` continues to
  support all current direct-light profiles and both shadow states; and
- returned `OpticalComputeResult` metadata and readiness/resource ownership
  remain compatible.

Compatibility may allocate, adapt host values, create events, and wait for
profiling. It uses the same shared kernels but cannot claim C4c1 native-stage
evidence or enter a benchmark lane. No fallback silently selects the old
kernel definitions.

Any numerical change outside the already frozen float32 tolerances is a C4c1
failure, not a new baseline.

## 14. Tests and acceptance gates

### 14.1 CPU-safe and structural tests

Tests must prove:

- CPU-only import remains valid without Warp;
- the direct-light kernel/device-function definitions have one production
  source;
- the ordered buffer ABI matches the formal output contract;
- diagnostics-off has zero overlay fields and diagnostics-on has exactly nine
  distinct fields in canonical order;
- each shadow/diagnostic combination selects one exact fixed shade entry;
- no shade ABI contains a runtime mode flag or dynamic counter collection;
- evidence strict round-trip and hash recomputation reject malformed or
  contradictory payloads; and
- neither the gate nor closed-worker runtime contract was activated.

### 14.2 Pre-launch negative tests

Before the first recorded launch, reject:

- wrong device, dtype, rank, shape, contiguity, count, or stream;
- default/ambient-device substitution for the explicit target;
- missing, duplicated, reordered, aliased, or overlapping output storage;
- nine-counter partial overlays or counter aliases;
- same-shape replacement scene/world/BVH/output arrays;
- same-content reupload instead of exact a2 allocation identity;
- wrong/copy/unregistered a2 readiness authority;
- scene/BVH contract or primitive/node-count mismatch;
- profile/shadow/diagnostics/entry mismatch;
- fixed-fixture/replay provenance mismatch;
- command deletion, insertion, replacement, or reordering;
- ordinary unseal/reseal or signature-baseline replacement; and
- `object.__setattr__()` drift of factory or bundle fields.

Every case asserts zero `Launch.launch` submissions and no completion/result
object. Command objects may already have been recorded during the untimed seal.

### 14.3 Real GPU correctness matrix

The GPU suite uses an asymmetric non-square camera with `fx != fy`, noncentral
principal point, nonidentity rotation, misses, planes, multiple triangles,
different material IDs, directional and point lights, and distinguishable
per-pixel outputs.

It directly checks:

- row-major origins and directions against host pinhole rays;
- every physical geometry, ID, material, RGB, intensity, primary diagnostic,
  and enabled overlay array against the compatibility oracle;
- formal exposed channels for all three output profiles;
- shadows false/true crossed with traversal diagnostics false/true;
- the four fixed shade entries actually used, not merely declared;
- no-shadow overlay counters remain independent zeros;
- first-hit tie-breaking and miss NaN/inf/zero values;
- ambient current CUDA device and stream restoration; and
- single-GPU plus target `cuda:1`/ambient `cuda:0` execution when two devices
  are available.

### 14.4 Reuse and contamination tests

For each shade variant, tests write non-default sentinels to every physical
buffer, execute the same recorded bundle, verify results, overwrite all buffers
with different sentinels, and execute it again. Results and counters must be
identical across runs. A zero-ray binding separately proves both scalar-reset
commands execute.

Runtime hooks around the recorded-launch leaf reject any observed allocation,
upload/materialization, NumPy conversion, Warp/Torch/DLPack bridge, module
load/compile, event operation, or host/device synchronization. Only the exact
pre-recorded `Launch.launch` operations are allowed. Preparation, compatibility
event handling, and correctness synchronization remain explicitly outside
that audited leaf.

### 14.5 Regression tests

Run the existing Warp optical runtime/profile/shadow tests, C3 recorded-command
tests, output-contract tests, CPU unit suite, and focused real-GPU suite. C4c1
produces no timing numbers, benchmark result JSON/JSONL, or baseline artifact.

## 15. Failure and lifecycle rules

Before any launch, validation failure is ordinary setup failure: no command is
submitted and the caller may discard the unexecuted bundle.

After the first launch begins, any launch exception poisons the entire bundle.
C4c1's blocking compatibility harness performs only test/oracle cleanup outside
the candidate path. It must not claim that a recorded event, host wait, or
successful synchronization is C4 lifecycle evidence.

C4c2 owns the real post-submit rule: it will extend the approved C4c0 guardian
to the complete slot/transaction graph and route any stage failure or transport
loss through the one acknowledged watchdog emergency-sync/quarantine path.
C4c1 neither duplicates that teardown state machine nor adds a second guardian.

## 16. Implementation slices inside C4c1

After this design is approved, implementation remains reviewable in three
bounded commits:

### C4c1a — source extraction and fixed kernel ABI

- create `optics/warp_native_render.py`;
- move shared raygen/first-hit/shadow/light mathematics;
- add deterministic initialization/reset entries and four shade variants;
- point compatibility executors at the single source; and
- pass existing executor parity/regression tests.

### C4c1b — caller-owned storage and recorded bundle

- add Warp a2/readiness authority validation;
- add buffers, camera binding, storage manifests, evidence, factory, bundle,
  compatibility driver, and pre-launch attack tests; and
- keep worker/gate/runner activation untouched.

### C4c1c — GPU parity and closure

- complete the four-way shade/profile matrix, direct ray checks, sentinel reuse,
  zero-ray reset, multi-GPU restoration, and contamination tests;
- update implementation/progress/reflection records; and
- obtain Claude approval before starting C4c2.

The split does not permit partially activating C4c1. Product GPU preparation
and every measured-sample submission remain disabled through all three commits.

## 17. Expected file surface

Expected additions/modifications are limited to:

```text
optics/warp_native_render.py
optics/warp_execution.py
optics/__init__.py                         [exports only if needed]
optical_worker/native_contract.py          [Warp authority symmetry only]
benchmarks/optical/native.py               [untimed adapter only, if needed]
tests/unit/optics/test_warp_native_render_contract.py
tests/gpu/test_warp_native_render.py
tests/unit/optics/test_warp_native_command_contract.py [extend as needed]
tests/gpu/test_optical_gpu_runtime.py                   [extend as needed]
collab/...C4c1 implementation note
PROGRESS.md
REFLECTIONS.md
```

Out of scope for modification:

```text
benchmarks/results/q54_p12_4/gate.json activation fields
benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json
optical_worker worker/supervisor protocol and dispatcher
optics/native_arena.py
optics/cuda_arena.py
benchmark runner/result writer/baselines
delivery/readback/B3/C0 publication
```

If implementation discovers that a listed out-of-scope file must change, C4c1
stops and returns to design review rather than broadening the slice silently.

## 18. Acceptance decision

C4c1 is complete only when all of the following are true:

1. one Warp source owns the renderer mathematics;
2. caller-owned buffers and every diagnostic array have exact validated
   storage;
3. all outputs are deterministic after arbitrary reuse sentinels;
4. four fixed shade ABIs contain no dummy or dynamic counter path;
5. actual recorded commands, arguments, stream, storage, and evidence remain
   inseparable under the negative matrix;
6. direct ray and complete output parity passes for profiles, shadows, and
   diagnostics;
7. the recorded launch leaf is clean under runtime instrumentation;
8. CPU-only import and existing compatibility behavior remain intact;
9. C4c0 worker trust/setup tests still pass; and
10. worker GPU preparation, C4c2-C4c4, C4d/B3, a3b, results, and baselines are
    still closed.

Approval of this document authorizes only C4c1a implementation. C4c1b and
C4c1c follow the bounded sequence above, and C4c2 requires its own review after
the entire C4c1 implementation is approved.
