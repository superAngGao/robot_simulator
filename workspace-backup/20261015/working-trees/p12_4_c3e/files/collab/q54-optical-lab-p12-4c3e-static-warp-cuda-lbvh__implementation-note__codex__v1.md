# Q54 Optical Lab P12.4c3e Static Warp/CUDA-LBVH Binding (Codex v1)

Date: 2026-09-03
Base: `2cb78ee689d9f3cc8fd6d3eb8f23e9ba553b0838`
Scope: C3e static Warp comparison binding and C3 closure only; no C4/B3,
a3b activation, benchmark result, baseline, or backend selection

## Outcome

C3e now has a temporary, fail-closed factory for the frozen Warp R2
`production_static -> cuda_morton_karras_lbvh_rebuild_v1` acceleration axis.
It preserves the runtime-native comparison boundary:

```text
Warp-owned replay/local scene
-> pre-recorded Warp world-scene/AABB update
-> CUDA LBVH command over setup-time Torch views of Warp-owned storage
-> Warp-owned BVH traversal view
```

The factory accepts only result-capable `canonical_native_replay_v1` Warp
input and only `production_static`. It waits for the a2 prepared-input event
outside timing, loads and prepares the CUDA LBVH extension outside timing, and
then seals concrete one-shot storage. C4 remains responsible for final arena
slots, generations, leases, ring depth, render/delivery binding, and any
result-capable evidence type.

## Ownership and stable-view ABI

`allocate_warp_cuda_lbvh_comparison_storage(...)` creates all LBVH outputs and
aligned CUB workspace as Warp-owned CUDA arrays. The sealed bundle creates
exactly 20 stable Torch views once, outside timing:

- 2 world AABB inputs;
- 17 CUDA LBVH outputs;
- 1 opaque workspace view.

The core LBVH API now exposes the narrow `CudaWorldAabbBuffers` input type, so
the temporary lane does not create unused Torch views of other Warp world-scene
arrays. Every view must have the same pointer and shape as its Warp owner. The
bundle strongly owns the Warp arrays, aligned workspace owner and slice, Torch
views, Warp stream, Torch `ExternalStream`, prepared input, factories, stages,
and traversal view for its full lifetime.

The Warp stream and Torch external stream must have the same indexed device
and same nonzero native handle. Both scene update and LBVH enqueue are scoped
through that external stream, which also restores a deliberately mismatched
ambient Torch device after each call.

## Timeline and completion ownership

`CudaLbvhRebuildStage.enqueue_bound_stream()` is a completion-neutral command
entry: it launches the already sealed fixed ABI on its bound stream and records
no event. The CUDA transaction path still uses the same bound launch helper and
retains its unique `finish()` event semantics.

The temporary Warp bundle is an explicit one-shot state machine:

```text
open -> arm (outside timing) -> scene_enqueued -> acceleration_enqueued
```

Out-of-order calls fail. Any enqueue exception poisons the bundle. C3e does not
create an intermediate ready event or a terminal event; the future Warp C4/B3
sequence owner must append render/delivery and record the one terminal
completion at the row endpoint.

## Provenance and fail-closed validation

The aggregate comparison evidence binds:

- the exact acceleration class, operation, topology origin/lifetime, device,
  and shared stream handle;
- native-input manifest, scene contract, replay digest, and frame index;
- the Warp scene-stage evidence digest;
- the CUDA LBVH-stage evidence digest;
- the ordered stable-view names and pointer/shape/dtype alias manifest;
- setup stable-view count and timed `bridge_creation_count=0`.

`arm()` rebuilds and compares the evidence and checks the entire object graph:
scene outputs are the exact CUDA AABB inputs; each Torch output/workspace view
aliases the expected Warp owner; the Warp traversal view uses the exact CUDA
LBVH output arrays; all resource owners remain retained; and no intermediate
event exists. Factory fields, stream identity, view ordering, provenance,
stage ABI, and traversal storage are immutable or rejected before first
enqueue if tampered with, including `object.__setattr__()` negative tests.

## Runtime audit and cleanup

The a3a source coverage manifest now names the real prepared Warp scene/refit,
temporary Warp static scene/LBVH, CUDA scene, and CUDA LBVH enqueue entries.
Source hashes remain coverage only. Runtime instrumentation remains the source
of timed evidence and covers the temporary comparison entry for extension
load, allocation, upload/materialization, synchronization, and Warp/Torch or
DLPack view creation.

The runtime-hook registry is versioned as
`q54.p12.4.runtime-hook-registry.v2` with SHA-256
`fd03c565975848414d47f40331e8131b1fd3950740f3f5ea04ddfed02b3467e9`.
The frozen gate records that exact registry identity; no performance or
capability threshold changed.

The cleanup ledger now enumerates the prepared Warp command surface, native
CUDA LBVH workspace/stage surface, temporary stable-view comparison binding,
and its GPU test. Terminal cleanup rules are unchanged: the temporary binding
is deleted with the losing lane, or retained only if the already-frozen
independent Warp/CUDA-LBVH benefit and maintenance gate authorizes
productionization.

## Verification

```text
python -m pytest -q tests/unit
1149 passed, 20 skipped

python -m pytest -q \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_cuda_lbvh_native.py \
  tests/gpu/test_optical_warp_cuda_lbvh_comparison.py \
  tests/gpu/test_cuda_optical_timeline.py
107 passed
```

The focused GPU test directly checks all 20 Warp/Torch pointer aliases, the
single native stream handle, zero observed timed bridge calls, ambient-device
restoration, mutation/order rejection, and Warp first-hit parity against the
existing CPU-BVH oracle.

The frozen Go2 asset is not present on this server, so no Go2 result or
baseline was attempted; the existing asset preflight remains fail closed.
Targeted `ruff check` and `git diff --check` pass. The a3b activation remains
`false` with a null activation commit, and no benchmark artifact was emitted.
