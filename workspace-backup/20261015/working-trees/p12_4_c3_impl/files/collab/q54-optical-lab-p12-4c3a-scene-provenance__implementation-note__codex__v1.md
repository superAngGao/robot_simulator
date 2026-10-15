# Q54 Optical Lab P12.4c3a Scene and Provenance Foundation

Author: Codex

Date: 2026-08-18

Status: implemented for Claude review; C3b-C3e, C4/B3, a3b results, baseline,
and performance inspection remain closed

Approved C3 design:
`435ffb275674c04f6452cb41775533db0f47f1d3`

## Outcome

P12.4c3a implements the CPU-safe logical scene contract and the complete
pre-enqueue provenance chain required by the approved C3 v4 design. The Warp
and Torch candidates now allocate directly and independently from one immutable
`PackedOpticalSceneHost`; neither candidate is materialized through the other
runtime. The existing Warp product builder consumes that same packed object, so
registry packing has one implementation rather than a benchmark copy.

This slice does not implement timed Warp refit, Torch scene-update kernels,
native CUDA LBVH rebuild, the static Warp/Torch comparison binding, final C4
slots, B3 delivery, or a result-capable lane. No benchmark sample, baseline, or
performance result was produced or inspected.

## Single logical scene materialization

`optics/scene_contract.py` adds `PackedOpticalSceneHost`. The module imports
neither Torch, Warp, nor benchmark code. It owns the only registry-to-array
packing implementation and freezes:

- source case/scene/materializer identity;
- one canonical semantic for every local geometry, metadata, material, light,
  role, source-order, and instance array;
- deterministic role bits, primitive order, degenerate-triangle filtering,
  dtype, shape, byte count, content digest, and aggregate scene digest;
- read-only, C-contiguous NumPy arrays.

`build_device_optical_scene()` now delegates to this contract. Warp and Torch
also expose explicit builders from a packed scene and zero-copy wrappers over
already prepared runtime-owned arrays. The wrappers validate every exact name,
dtype, shape, layout, device, and scene identity and strongly retain the
prepared owner. They do not allocate or create Warp/Torch interop views.

## Replay and fixed R0 ownership

`CanonicalTransformReplayHost` owns the immutable canonical dynamic transform
sequence. Its digest covers case identity, sequence contract, frame count,
dtype, shape, and contents. Per-sample evidence binds both the full replay
digest and selected frame index. `prepared_transform_frame_signatures()` binds
that frame to the original runtime allocation plus its exact active byte range;
it does not create a selected-frame copy.

`CanonicalFixedRenderFixtureHost` can be constructed only from the exact
`PackedOpticalSceneHost` object. It owns only derived world geometry, AABBs, and
normalized CPU-median BVH arrays. Metadata, materials, lights, roles, and local
geometry stay owned by the packed scene. Names and semantic owners must be
disjoint. `CanonicalNativeInput.from_scene_contract()` uploads the union once
per candidate and retains the source object identities; an equal-content scene
object cannot substitute for the fixture's owner.

The generic `CanonicalNativeInput.from_arrays()` remains available for
diagnostic a2 fixtures but has null scene provenance and cannot enter a
result-capable wrapper or runner submission.

## One-way evidence DAG and exact dataflow

The native-input schema is v2, acceleration receipt schema is v2, gate/result
schemas are v7, and row-semantics inputs are v4. The frozen gate includes the
approved C3 design identity and explicitly records that no a3b result-capable
binding exists before C4/B3 review.

The immutable dependency order is:

```text
packed scene + replay/fixed fixture
-> independently prepared runtime storage
-> independent stage evidence
-> NativeExecutionBindingEvidence
-> ExecutionReceipt/input_provenance_sha256
```

`NativeStageEvidence` has no self hash and cannot refer to the outer binding.
The outer binding alone stores each independently computed stage digest. The
acceleration receipt may refer to the independent acceleration-stage digest,
but not to the aggregate binding.

Storage signatures include runtime owner, backend, device, base allocation,
dtype, shape, allocation bytes, and active byte range. The binding validator
checks actual arrows rather than trusting an aggregate hash:

- R0 render reads exactly the prepared packed-scene plus derived-fixture union;
- R1/R2 scene update reads the exact local scene and selected replay ranges;
- acceleration inputs are exact outputs of scene update;
- render inputs are exact prepared or preceding stage outputs and include both
  updated scene and acceleration storage;
- later stages cannot introduce an unbound same-shaped allocation.

Before the timed context begins, the runner independently reconstructs the
prepared scene, fixture, and selected replay storage signatures and compares
them with the lane-generated binding. A mismatch fails before `submit_sample()`
and leaves the operation trace empty. The receipt must return the identical
validated binding and digest.

## Native ownership proof

The GPU regression forbids `wp.to_torch`, `wp.from_torch`, and both Torch DLPack
entry points. It then creates the result-capable input twice from the same
logical source: once as Warp-owned CUDA allocations and once as Torch-owned
CUDA tensors. Non-empty allocations must have distinct addresses. The Warp and
CUDA local-scene wrappers retain the exact respective a2 objects, and every
wrapped scene array is the same runtime object prepared for that lane.

Zero-length canonical arrays are valid typed contract members and are not
misreported as failed allocations merely because their runtime pointer is
null. They still carry deterministic empty-storage identity in provenance.

## Cleanup and phase boundary

The P12.4e cleanup ledger now covers the durable logical scene contract,
selected-backend replay/fixed-fixture/binding contracts, prepared Warp scene
surface, and candidate CUDA scene surface. If CUDA wins, Warp optical scene/BVH
ownership is removed after migration. If Warp wins, the Torch CUDA scene
surface and candidate-only tests are deleted. The logical scene contract and
selected single-lane regression core remain in either branch.

Current lane adapters still expose only preflight. C3a supplies schemas and
one-shot test bindings, not a product submission API. C3b-C3e must implement
and review their real stages; C4 must create final slot/generation/lease-bound
bundles; B3 must close delivery. Until those reviews pass, a3b and every result
or baseline output remain unauthorized.

## Verification

Commands run on `192.168.80.101` in the existing `robot_sim` environment:

```text
python -m pytest -q \
  tests/unit/optics/test_scene_contract.py \
  tests/unit/optics/test_device_scene.py \
  tests/unit/optics/test_registry_builder.py \
  tests/unit/benchmarks/test_optical_benchmark_c3a.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py

python -m pytest -q tests/unit/optics tests/unit/benchmarks

python -m pytest -q \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_cuda_optical_timeline.py

ruff check benchmarks/optical optics/scene_contract.py optics/cuda_scene.py \
  optics/device_scene.py optics/device_bvh.py \
  tests/unit/benchmarks/test_optical_benchmark_c3a.py \
  tests/unit/optics/test_scene_contract.py \
  tests/gpu/test_optical_benchmark_native.py

git diff --check
```

Observed:

- focused C3a/contract/a3a unit suite: 79 passed;
- full optics plus benchmark unit suites: 359 passed;
- affected optical GPU suites: 62 passed;
- Ruff: passed;
- `git diff --check`: passed after LF normalization.

## Review request

Please review whether:

1. `PackedOpticalSceneHost` is now the sole registry materialization source and
   preserves all prior Warp product semantics;
2. R0 shared versus derived array ownership is disjoint, single-upload, and
   bound to exact packed-object identity;
3. replay hash/index and selected-frame active ranges cannot be replaced by an
   equal-shaped allocation before enqueue;
4. stage/binding/receipt hashes form a strictly one-way DAG with no self or
   aggregate back-reference;
5. dataflow validation mechanically connects prepared scene/replay/fixture,
   scene-update, acceleration, render, and later storage;
6. Warp and Torch local scenes are independently runtime-owned with no bridge
   or hidden duplicate upload;
7. schema/gate changes occurred before any result and still keep a3b closed;
8. the cleanup ledger yields one clean selected backend in either terminal
   branch.
