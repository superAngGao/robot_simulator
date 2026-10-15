# Q54 P12.4 C3b prebound Warp scene/refit implementation note v1

Status: implemented for Claude review. This slice covers only the approved C3b
Warp scene-update/refit data plane. C3c-C3e, C4/B3, a3b result capability,
baseline generation, and backend selection remain closed.

## Outcome

C3b replaces per-frame allocating/list-built Warp scene update and BVH refit
calls with setup-time factories and immutable one-shot command bundles:

```text
exact a2 Warp scene + exact replay-frame view
  -> caller-owned world geometry/AABB outputs
  -> setup-time record_cmd scene bundle

CPU median leaf-4 topology from the setup frame
  -> immutable uploaded topology + caller-owned mutable bounds
  -> setup-time record_cmd refit bundle

timed path
  -> scene Launch.launch(frozen stream)
  -> leaf/internal Launch.launch(frozen stream)
```

The native enqueue methods do not allocate output arrays, construct Warp
launch arguments, create or wait on events, synchronize, or call `wp.launch`.
They only submit the commands recorded and sealed before timing on the exact
prepared stream.

## Caller-owned scene outputs

`optics/device_scene.py` adds:

- `WarpSceneTransformInputs`, an exact frame view over the already uploaded a2
  replay allocations with replay hash/index and strong source ownership;
- `WarpWorldSceneBuffers`, the caller-owned world triangle, plane, and triangle
  AABB outputs;
- `WarpSceneUpdateCommandFactory`, which validates the scene, transforms,
  output storage, device, stream, shape, dtype, contiguity, capacity, and full
  input/output disjointness before recording;
- `WarpSceneUpdateCommandBundle`, an immutable fail-closed bundle whose first
  launch failure poisons all later enqueue attempts;
- an explicit setup/test compatibility allocator. It is not a C4 arena and
  does not confer product lifecycle or result capability.

The scene dispatch plan contains at most one triangle world/AABB command and
one plane command. Empty primitive classes create no dummy launch. Evidence
binds the actual manifest, scene, replay frame, device, stream, exact input and
output storage manifests, dispatch count, and ordered dispatch-plan digest.

For zero-body static replays, Warp 1.12 cannot slice through a later zero-sized
dimension. The typed replay view therefore retains the exact base allocation
and selects the frame as `env_idx`; static instances never dereference body
transforms. No substitute allocation is created.

## Immutable topology and slot-composable refit

`optics/device_bvh.py` adds a canonical hash of the exact normalized CPU
topology arrays before their one-time upload. `DeviceOpticalBvh` carries both
the source scene-contract hash and this actual BVH topology hash.

`WarpBvhRefitCommandFactory` owns the immutable topology facts and prepared
stream, not a final C4 output slot. `seal(world=..., bvh=...)` accepts a
concrete caller-owned bounds bundle and requires it to share the factory's
exact immutable topology storage. It rejects:

- a different scene contract or topology digest;
- replacement of any topology or primitive-metadata allocation;
- different topology counts, depth, leaf size, or level ranges;
- wrong-device, wrong-dtype, wrong-shape, non-contiguous, or null AABB/bounds
  storage;
- overlap between mutable node bounds and any world-scene, AABB, topology, or
  primitive-metadata allocation.

This makes the C3 factory reusable by C4 when C4 creates each final arena slot:
each slot supplies distinct mutable bounds, while all sealed bundles retain the
same immutable topology allocations. A sealed bundle has no rebind method.

The refit plan records one leaf command followed by the exact non-empty
internal levels from deepest to root. Empty BVHs record zero commands. Evidence
keeps the a2 source-topology digest separate from the actual CPU-median BVH
topology digest and binds both without a self-reference.

## Exact a2 helpers and compatibility boundary

`benchmarks/optical/native.py` now provides narrow helpers that derive the
Warp local scene, replay-frame view, scene factory, and refit factory from the
same result-capable a2 `WarpPreparedNativeInput`. They reject wrong input
contracts, frame ranges, primitive counts, devices, and scene-contract
provenance before command recording.

Existing allocating scene/BVH builders and executor facades are preserved for
compatibility and correctness-oracle use. Their upload, allocation, event, and
synchronization work stays outside native timing and they remain benchmark
ineligible.

## Tests

New CPU/source-contract tests prove:

- timed bundle entry points contain only frozen `Launch.launch` submission;
- factories use `record_cmd=True` and `record_tape=False` during setup;
- C3b evidence cannot claim C4 slot, generation, lease, ring, or result fields;
- scene output-disjointness coverage tracks every formal packed-scene array;
- launch failure poisons immutable bundles.

New real-GPU tests prove:

- exact canonical a2 scene/replay provenance and distinct source/actual
  topology evidence;
- real timed scene-update/refit submission succeeds while `wp.launch` is
  replaced with a failing sentinel;
- operation order is exactly scene update then acceleration and the runtime
  audit observes no forbidden operation;
- world geometry/AABBs match the allocating compatibility path;
- leaf and multi-level internal refit bounds match the existing refit oracle,
  topology arrays/hashes remain unchanged, and traversal matches a fresh BVH;
- a factory seals a distinct caller-owned bounds slot while rejecting replaced
  immutable topology storage;
- aliases, wrong AABB dtype, wrong scene provenance, stream drift/rebind, and
  post-failure reuse fail closed;
- triangle-free/plane-only input emits one scene command and zero refit/dummy
  commands.

Verification at review handoff:

- targeted Ruff checks: passed;
- C3b unit and real-GPU tests: 9 passed;
- benchmark/optics/optical-GPU regression: 445 passed;
- full unit suite: 1138 passed, 20 skipped (one existing pytest deprecation
  warning).

## Gates intentionally unchanged

- The frozen C4/B3 activation remains false.
- No C4 result-capable binding constructor exists.
- a3b cannot submit a result-capable sample.
- No benchmark result, baseline, performance observation, or backend decision
  is created.
- Permanent a3a hook-manifest and cleanup-ledger integration remains assigned
  to the approved C3e closure slice.
