# Q54 Optical Lab P12.4c2 Native CUDA Execution Timeline

Author: Codex

Date: 2026-08-14

Status: design request for Claude review; this commit does not authorize C2
implementation or any benchmark sample

Approved parent contracts:

- P12.4 backend parity/performance gate v5;
- P12.4a3 native-lane dependency order v3;
- P12.4a3a eligibility/runtime-instrumentation implementation;
- P12.4c0a CUDA publication binding v3;
- approved P12.4c0b implementation at `706ab8513da6cad99c23385dc975b877696abba9`.

## Outcome

C2 establishes one production, Torch-native CUDA execution timeline shared by
the later CUDA scene/LBVH, render, pack, and delivery work. It makes the current
CUDA raygen/first-hit/direct-light kernels usable through caller-owned output
buffers on an explicit indexed Torch stream and returns a native completion
event without a host wait.

C2 does **not** claim that scene update, LBVH, pack, delivery, or the benchmark
lane is native yet. Those adopters belong to C3, C4, and B3. C2 is complete as
an independently reviewed implementation slice when the common timeline and
the CUDA render adopter pass the tests below. The full C2 prerequisite for a3b
is not closed until C3 and B3 prove their operations use the same timeline.

This distinction prevents two bad outcomes:

- fake scene/LBVH/pack shims created only so C2 can claim broad coverage;
- separate asynchronous contracts that later require another migration before
  a fair native benchmark can run.

No R0/R1/R2/S result, baseline, backend selection, product-default change,
profile optimization, or kernel fusion is authorized in this slice.

## Current code and the gap

`optics/cuda_direct_light.py` already has several correct low-level properties:

- raygen is out-style;
- each CUDA entry uses `c10::cuda::CUDAGuard`;
- each kernel obtains
  `at::cuda::getCurrentCUDAStream(anchor.get_device())`;
- camera rays are flat row-major and have direct host parity tests.

It is not a native asynchronous transaction:

- first-hit and both shade entries allocate output tensors in C++;
- the Python executors allocate ray buffers per call;
- Warp scene/BVH channels are converted through `channel_to_torch()` per call;
- Warp readiness is resolved with `wp.synchronize_event()` on the host;
- first-hit calls `torch.cuda.synchronize(device)` and shade calls it again;
- timing events are created during the call and resolved only because those
  blocking synchronizations have already occurred;
- the public result has `ready_event=None` and cannot participate in a native
  downstream wait or an output-slot lease.

`optics/cuda_lbvh.py` and `tools/optical_pipeline_lab/rgb_pack.py` also allocate,
bridge, or synchronize. C2 does not wrap those paths and call them native. C3
replaces the scene/LBVH transaction; B3 replaces CUDA RGB8 pack and delivery.

## Frozen terminology

The following terms are distinct:

- **compatibility facade**: the existing blocking
  `CudaDeviceBvh*OpticalExecutor` API over Warp-owned scene/BVH objects. It may
  prepare views and allocate outside the native submission, then wait on the
  returned terminal event. It remains benchmark-ineligible.
- **native render submission**: only the operations enqueued by the new
  render binding on its explicit Torch stream. It performs no allocation,
  upload/materialization, Warp/Torch view creation, extension load, or host or
  device-wide synchronization.
- **native transaction**: in later slices, scene update through acceleration,
  render, pack/delivery, and completion on the same timeline. C2 supplies the
  timeline and render portion, not the complete transaction.
- **completion recorded**: a CUDA event has been successfully recorded after
  the final operation on the selected stream. It does not mean the GPU has
  physically completed the work.
- **completion resolved**: an external consumer has established a legal wait
  or, outside a timed submission, performed a terminal host wait.

## Module ownership

The runtime-neutral timeline lives in a new production module,
`optics/cuda_timeline.py`. Scene/LBVH and pack must not depend on the
direct-light renderer merely to share stream semantics.

`optics/cuda_timeline.py` owns:

- explicit device/stream identity and scoped current-stream selection;
- completion-event validation and recording;
- ambient device/current-stream restoration;
- optional deferred diagnostic-event recording;
- immutable completion/profile tickets.

`optics/cuda_direct_light.py` owns:

- render input/output buffer contracts;
- extension preparation;
- bound first-hit and direct-light submissions;
- compatibility facade adaptation;
- raygen, traversal, and shade extension code.

C3 will make its scene/LBVH entries accept the same execution context. B3 will
do the same for pack/delivery. C4 will own actual reusable buffer, scratch,
completion-event, and output-lease pools.

The production modules remain importable on CPU-only hosts. CUDA dependencies
are required only when constructing/preparing CUDA objects.

## Explicit execution context

The durable context has the conceptual shape:

```python
context = CudaOpticalExecutionContext(
    device=torch.device("cuda:1"),
    stream=torch.cuda.Stream(device="cuda:1"),
)
```

The exact class spelling may change during implementation, but these semantics
are frozen:

1. `device` must be an indexed CUDA `torch.device`; ambient current device is
   never used as the target.
2. `stream` must be a live Torch CUDA stream whose indexed device equals the
   target. The context holds a strong reference to the stream owner.
3. Raw integer stream handles are not accepted by render callers. The C0
   publication binding may read this owner's handle for its approved
   physics-producer handoff.
4. Each submission enters both `torch.cuda.device(target)` and
   `torch.cuda.stream(selected_stream)`, records all work there, and restores
   the caller's ambient current device and per-device current streams on every
   success and failure path.
5. CUDA C++ entries receive no raw stream argument. Their tensor anchor selects
   the guarded device, and they launch on the indexed PyTorch current stream.
6. Context construction does not load or compile extensions. The renderer has
   an explicit `prepare()` step outside any timed boundary and a native submit
   fails closed if preparation did not finish.
7. The context exposes no host/global synchronize helper. A compatibility or
   test caller may wait on a returned event only outside native submission and
   outside benchmark timing.

The context is the only stream-selection mechanism for the later C3 and B3
native operations. They may enqueue device-side waits on this stream, but may
not replace it or silently use a default stream.

## Native-ready input rule

C2 render inputs are Torch CUDA tensors already resident on the target device.
All input views, uploads, extension compilation, validation fixtures, and
output/event allocation happen before native submission.

The render API accepts no Warp arrays, Warp events, DLPack objects, NumPy
arrays, host tensors, or generic `ready_event` object. It does not call
`channel_to_torch()`, `wp.to_torch()`, `wp.from_torch()`, or any host wait.

For C2 tests, inputs may be independently prepared Torch-owned tensors or
prebound compatibility views prepared outside submission. For a3b, C3 must
supply Torch-owned scene/LBVH storage and enqueue its updates on the same
stream. For S, C0 must first enqueue its publish-event wait on that stream.
Those later input contracts do not change C2 render semantics.

## Prepared and bound render entry

Extension load/JIT and structural validation are split from submission:

```text
prepare renderer/extension              outside timing
bind stable inputs, outputs, constants  outside timing
submit                                  timed in later a3b
resolve/inspect                         outside primary timing
```

A bound render operation strongly retains:

- the execution context and prepared extension module;
- every input tensor used by raygen/first-hit/shade;
- every caller-owned output/diagnostic tensor;
- immutable camera, lighting, role-mask, shadow, and output-profile scalars;
- a stable tensor-storage signature used to reject resize/replacement.

Binding performs full device, dtype, shape, contiguity, count, capacity, and
storage-overlap validation. Submission performs only non-synchronizing identity
and storage-generation checks needed to fail closed if a caller mutated the
bound objects. C4 later makes this mutation impossible while a slot is leased.

Outputs are disjoint from inputs and from one another except where an explicit
read-only alias is documented. No diagnostic output aliases another counter.
The old no-shadow behavior of returning nine references to one zero scalar is
not part of the native contract.

The native renderer supports two bound operations:

- first-hit over pre-existing rays, for low-level geometry parity;
- camera direct-light: raygen -> first-hit -> optional shadow shade.

It supports the currently declared direct-light profiles and both shadow
states. Profile filtering is metadata/output exposure; it cannot trigger an
allocation inside submission. All intermediate geometry required for shade is
therefore present in the caller-owned render bundle even for RGB-only profiles.

## Caller-owned output contracts

The C2 render bundle contains preallocated contiguous Torch CUDA tensors on the
context device:

| Group | Outputs |
| --- | --- |
| rays | `origins`, `directions` |
| geometry | `hit_mask`, `range_m`, `position_world`, `normal_world`, `numeric_instance_id`, `material_index` |
| primary traversal diagnostics | `bvh_stack_overflow_count`, `bvh_max_stack_depth` |
| shade | `rgb`, `intensity` |
| primary shadow diagnostics | `shadow_stack_overflow_count`, `shadow_max_stack_depth` |
| optional traversal overlay | the nine canonical shadow traversal scalar channels |

Per-ray shapes and dtypes continue to follow the formal optical output
contract. Primary and traversal diagnostics use distinct contiguous `[1]`
`int32` tensors. The optional nine-counter overlay is either completely absent
or completely present and agrees with the diagnostics flag.

C2 may provide an allocation helper solely for tests and the blocking
compatibility facade. That helper is never called by native submission and is
not an arena. C4 replaces benchmark preparation with its bounded reusable arena
and lease contract.

## Out-style extension contract

The extension surface after C2 is:

```text
pinhole_camera_rays_out(..., origins_out, directions_out)
first_hit_rays_out(..., geometry_outputs..., primary_counters...)
shade_direct_light_no_shadow_out(..., shade_outputs..., counters...)
shade_direct_light_with_shadows_out(..., shade_outputs..., counters...)
```

Argument order may follow the current extension convention, but each entry must
obey these rules:

- it allocates no Torch tensor, CUDA buffer, event, or scratch storage;
- it validates every input/output tensor's CUDA device, dtype, rank, shape,
  contiguity, and related lengths;
- it uses an output tensor on the target device as its guard/stream anchor and
  requires all tensors on that exact device;
- it obtains `at::cuda::getCurrentCUDAStream(anchor.get_device())` only after
  installing `c10::cuda::CUDAGuard`;
- all initialization, memsets, and kernels use that same stream;
- it performs the standard CUDA launch check but no host wait, event query, or
  device-wide synchronize;
- allocating `first_hit_rays` and shade extension exports are removed from the
  native surface rather than retained as a hidden fallback.

The first-hit kernel must explicitly overwrite every per-ray output for both
hit and miss. Miss values preserve current behavior: zero hit flag and IDs,
infinite range, and NaN position/normal. It may not rely on prior buffer
contents or a preceding allocating `torch::full()`.

Primary counters and enabled overlay counters are zeroed with same-stream
`cudaMemsetAsync` before their kernels. This initialization also runs for an
empty ray set where no kernel is launched. Shade kernels overwrite every RGB
and intensity element. No-shadow diagnostics, when requested, are explicit
zero outputs rather than aliases.

Extension naming/versioning is bumped so stale compiled code cannot satisfy the
new signature.

## Completion and ownership

Native submission receives a CUDA completion event prepared and initialized on
the target device outside timing. In C2 tests it is caller-owned. C4 later owns
event slots and their reuse lifecycle.

After the final selected operation, submission records that event on the exact
execution stream and returns an immutable completion ticket containing:

- the same event object, held strongly;
- target device ordinal;
- selected stream identity/handle evidence;
- operation/profile/shadow identity;
- strong references to the bound inputs/outputs required until downstream wait
  or terminal resolution;
- optional deferred profiling evidence.

No event is constructed, lazily initialized, queried, elapsed-time-resolved, or
host-synchronized inside submission. A ticket is returned only if all immediate
launch checks and terminal event record succeed. It says `recorded`, never
`physically_complete`.

A same-device downstream Torch stream may use CUDA stream-wait-event semantics.
A blocking compatibility caller may synchronize that one terminal event after
submission. Neither action changes the native submit trace. Event reuse,
double-release, active-close, exception recovery, and final drain are C4
responsibilities; therefore C2 alone cannot become a benchmark lane.

For the later S path, the C0 consumer completion event remains physics-owned.
The optical completion ticket is not inserted into `PublishedRing`. The last
physics-buffer reader is the C3 scene update, so C0 may record its own done event
there while the separate optical output lease continues through render and
delivery.

## Timeline and synchronization

For a camera direct-light submission, the C2 render portion is exactly:

```text
selected Torch stream current
  -> optional diagnostic start records
  -> raygen out
  -> first-hit counter zero + first-hit out
  -> shade counter zero + shade out
  -> optional diagnostic end records
  -> terminal completion event record
return without host wait
```

There is no first-hit synchronization between traversal and shade. Stream order
provides the dependency. There is no default-stream assumption.

Later native transactions extend the same stream on each side:

```text
C3 scene update -> C3 acceleration -> C2 render -> B3 pack/delivery -> completion
```

C2 must not add placeholders for the missing portions. C3 and B3 acceptance
tests must demonstrate exact context/stream identity and one ordered trace.

## Profiling semantics

Primary benchmark runs request no phase timing and contain only the terminal
completion event required by the transaction.

Diagnostic profiling is a separate pass. The caller supplies pre-created,
pre-initialized CUDA timing-event pairs for the requested phases. Submission
only records them on the selected stream and returns a deferred profile ticket.
`elapsed_time()` is read only after terminal completion and outside the native
submission/timed boundary.

The compatibility facade may continue to fill the current `render_profile`
list, but it does so after its terminal event wait. That blocking facade remains
ineligible and its values are not primary replacement-gate evidence. C2 does
not preserve CPU launch-wall-time fields by inventing asynchronous meanings.

## Compatibility facade

The existing `CudaDeviceBvhOpticalExecutor` and
`CudaDeviceBvhDirectLightOpticalExecutor` remain available during the selection
phase so current product tests and the Warp oracle continue to work.

They may:

- wait for current Warp-owned input readiness;
- create Warp/Torch views;
- visibly allocate a one-call render bundle and completion/profile events;
- invoke the new bound native render operation;
- wait on the returned terminal event;
- return the current blocking `OpticalComputeResult(ready_event=None)`.

They may not call the removed allocating C++ entry points. Target-wide
`torch.cuda.synchronize()` is removed; a compatibility wait is scoped to the
returned terminal event. Output values, channels, profile filtering, and
diagnostic-overlay behavior remain unchanged.

This facade is explicitly not an a3b lane: allocation, bridge creation, Warp
host wait, and terminal host wait remain visible to runtime instrumentation.
If CUDA wins, C5/e removes or rewires the compatibility-only path according to
the selected product architecture. If Warp wins, the native CUDA renderer and
its temporary adapter are deleted.

## Runtime instrumentation and eligibility

C2 registers the real low-level render submission entry in the a3a source
coverage/runtime-hook plan. The source hash remains only a coverage manifest.
Runtime wrappers remain the evidence.

Tests execute the actual native submission under the timing guard and prove:

- its high-level render operation is observed from the wrapped callable;
- no forbidden allocation, upload/materialization, synchronization,
  Warp/Torch/DLPack bridge, staging, or extension-load hook fires;
- deliberately moving allocation, `channel_to_torch()`, extension load, or a
  host wait inside the real submit is caught by the existing bottom-level hook,
  even if the high-level entry does not self-report it.

The temporary CUDA lane adapter remains preflight-only and ineligible after C2.
The phase authorization map remains empty for a3a. No `submit_sample()` is
added, and no schema-valid result can be written. C3/C4/B3 and a3b still gate
eligibility.

## Failure behavior

Validation failure occurs before any kernel is enqueued. Immediate extension
launch or completion-record failure produces no completion ticket and marks the
caller-provided output/event slot unusable until its later C4 lifecycle decides
the disposition.

C2 does not invent product recovery for asynchronous CUDA-fatal errors. The
blocking compatibility facade surfaces them at its terminal event wait. The
future shipping composition routes publication/guardian failures through the
approved C0 poison contract and output-arena failures through C4. No failure
path may pretend completion, reuse a possibly in-flight buffer, or fall back to
the Warp optical lane.

## Acceptance tests

### CPU/unit tests

- CPU-only imports do not require Torch CUDA/Warp initialization.
- device must be indexed CUDA; CPU, unindexed, and stream/device mismatch fail.
- bound tensor contracts reject wrong device, dtype, rank, shape, contiguity,
  capacity, diagnostics-set mismatch, storage overlap, resize, and replacement.
- unprepared extension/runtime submission fails closed.
- completion/profile tickets distinguish recorded from physically completed.
- a3a authorization remains empty and the CUDA adapter remains ineligible.
- source coverage resolves the new real submit entry and does not relabel the
  compatibility facade as native.

### Single-GPU tests

- out-style raygen, first-hit, shadow-off shade, and shadow-on shade match the
  approved host/Warp oracle for all current profiles.
- direct origins/directions parity uses a non-square camera, non-central
  principal point, unequal focal lengths, and non-identity rotation.
- output buffers prefilled with sentinels prove every hit/miss/per-ray/scalar
  field is overwritten; empty-ray calls still reset counters.
- diagnostics overlay on/off has the formal channel set, distinct storage, and
  exact zero/nonzero behavior.
- a non-default explicit stream with an earlier producer operation proves
  kernels execute in order on that stream rather than the default stream.
- a second Torch stream waits on the returned event device-side and observes
  completed outputs without host synchronization.
- monkeypatch/trace tests reject any `torch.cuda.synchronize`, event/stream host
  synchronize, query, Warp wait, bridge, allocation, upload, or extension load
  within native submission.
- caller ambient device and current stream are unchanged after success,
  validation error, immediate launch error, and event-record error.
- deferred timing events resolve only after the external terminal wait and
  report the requested phase names; the primary path creates no timing events.
- compatibility executor output/channel/profile behavior remains unchanged and
  uses only a terminal event wait, while instrumentation still classifies that
  facade as ineligible.

### Multi-GPU tests, when at least two devices exist

Immediately before each low-level raygen, first-hit, shadow-off, shadow-on, and
full chained submission:

```python
torch.cuda.set_device(0)
assert torch.cuda.current_device() == 0
# context and every tensor remain on cuda:1
ticket = bound.submit(...)
assert torch.cuda.current_device() == 0
```

The test also installs distinct current streams on both devices and verifies
both are restored. Wrong-device input, output, stream, or event is rejected
before launch.

### Regression suites

- all existing CPU/unit tests;
- all existing GPU optical tests;
- focused a3a instrumentation/isolation tests;
- `git diff --check` and a clean committed worktree.

No benchmark performance number is read or written in C2.

## Implementation slices and review boundary

C2 implementation is one bounded code slice after this design is approved:

1. add `optics/cuda_timeline.py` and its CPU/unit tests;
2. convert the direct-light extension to out-only entries and bump its version;
3. add prepared/bound native render APIs and GPU correctness/stream tests;
4. route the blocking compatibility facade through the new out/timeline path;
5. extend a3a coverage and negative instrumentation tests without enabling a
   lane;
6. add a C2 implementation note, run the agreed suites, commit, and request
   Claude review.

Out of scope and forbidden in this slice:

- Torch-owned scene update or CUDA LBVH rebuild/refit;
- Warp refit changes or stable Warp-renderer views;
- reusable benchmark arenas, scratch planning, output leases, or final drain;
- RGB8 pack, D2H, async delivery, or ring policy;
- C0 shipping binding activation;
- benchmark coordinator, paired runs, samples, baseline, threshold, selector,
  fallback, or product default;
- new renderer math, fused kernels, profile-specialized kernels, SAH/builder
  tuning, or performance-driven gate changes.

## Later closure criteria

C2's common contract is a dependency, not a permanent extra backend layer.

- C3 must show scene update and acceleration enqueue on the identical context
  stream and return without host wait.
- C4 must provide complete preallocated storage/event ownership and leases for
  the frozen R0-R2 rows.
- B3 must show pack/delivery extend the identical timeline and terminate in the
  row-specific completion/reclaim endpoint.
- a3b may become eligible only when its full successful timed trace contains
  the frozen operations and zero contaminations.
- P12.4e deletes the losing renderer lane and all selection-only adapters. The
  selected backend retains only the common benchmark schema/workloads and its
  production execution path.

## Review questions for Claude

1. Does separating the independently reviewable C2 render adopter from the
   later full C2 prerequisite correctly avoid both fake shims and a second
   asynchronous contract?
2. Is caller-prepared completion-event ownership sufficiently bounded before
   C4 introduces pooling and leases?
3. Are the compatibility facade and native render submission separated tightly
   enough that runtime instrumentation cannot mistake the former for an
   eligible a3b lane?
4. Are any current output defaults, diagnostics, or event/timing semantics
   missing from the out-style conversion?
