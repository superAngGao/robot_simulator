# Q54 Optical Lab P12.4a2 Durable Benchmark Core — Implementation Note

Author: Codex

Date: 2026-08-11

Status: implementation candidate for Claude review

Approved design: P12.4 proposal v5 (`46a5d33`)

Parent implementation: P12.4a0/a1 (`231a54140ce188939a8ea60416baa3a409ff090f`)

## Scope

This slice implements only P12.4a2:

- durable backend-neutral single-lane submission and receipt validation;
- independent Warp-array and Torch-tensor CUDA input preparation;
- direct untimed native input parity and allocation-independence checks;
- explicit compile/warmup readiness evidence;
- timed-region contamination guards;
- canonical atomic benchmark artifacts.

It does not implement real R0/R1/R2 renderer lanes, the paired comparison
coordinator, baseline observation, a product backend selector, physics
publication binding, asynchronous CUDA delivery, or a replacement decision.

## Implemented contracts

### Independent native input

`CanonicalNativeInput` freezes C-contiguous logical host arrays, per-array
semantics, and the canonical case hash. Its manifest records dtype, shape,
bytes, per-array content hashes, aggregate content identity, and the ordered
fixed-topology identity.

`prepare_warp_native_input(...)` and `prepare_cuda_native_input(...)` each copy
directly from the logical host source into a new allocation owned by that
runtime on the requested CUDA device. They do not call `wp.to_torch`,
`wp.from_torch`, or DLPack. Producer event recording and synchronization happen
before timing. Ambient Torch device restoration is tested, including a target
device distinct from the ambient device when two GPUs are available.

`validate_prepared_native_input(...)` stages each lane independently outside
timing and compares actual values, dtype, shape, contiguity, content hash, and
topology hash. `assert_independent_native_inputs(...)` additionally rejects a
shared device address.

### Native readiness and timing

`NativeReadinessEvidence` binds the prepared manifest/content/topology to the
runtime lane, device ordinal, producer-ready event, direct validation,
allocation, compilation, and completed warmup count. The runner rejects fewer
than the gate's 10 frozen warmup frames.

`native_timed_submission(...)` covers only lane submission/enqueue. A lane must
return a pending sample, and `resolve()` occurs after the timed context exits.
The shared forbidden-operation set covers allocation, compilation, host upload
or materialization, global device synchronization, cross-lane wait, and both
Warp/Torch bridge directions. The future a3 concrete lane tests must instrument
their actual runtime entry points against this guard.

### Receipt ownership and artifacts

Result schema/gate v4 adds input manifest, content, topology, full readiness
evidence, and readiness hash to the lane-produced `ExecutionReceipt`. The
single-lane runner cross-checks them against the object actually submitted;
the standalone validator verifies internal consistency and the frozen warmup
count. The coordinator still adds only run/repeat/sample identity and timing.

`BenchmarkArtifactWriter` emits canonical JSON/JSONL with `allow_nan=False`,
atomic same-directory replacement, root escape rejection, duplicate-path
rejection, byte counts, record counts, and SHA-256 manifests.

## Repository cleanliness

The cleanup ledger now names the a2 modules as a durable single-lane benchmark
core retained in either terminal branch. No paired selector exists in this
slice. When a3 adds comparison-only orchestration or two renderer lanes, those
must receive separate P12.4e cleanup entries; the durable core must collapse to
the selected backend's regression lane without retaining a product fallback.

Production modules do not import `benchmarks.optical`.

## Validation

Targeted candidate validation in the `robot_sim` environment:

- benchmark contract/core and GPU native preparation: `33 passed`;
- Warp/Torch/DLPack bridge functions replaced with failures during both native
  preparation paths;
- direct values/layout/hash parity and independent device addresses checked;
- timed submit before resolve ordering and context restoration checked;
- wrong request, wrong input provenance, incomplete warmup, bad artifact hash,
  path escape, duplicate artifact, and NaN artifact failures checked.

Final repository validation used the existing environment with
`CUDA_HOME=/usr/local/cuda-12.8` and CUDA 12.8 first on `PATH` (the host's
unqualified `/usr/bin/nvcc` is CUDA 11.5 and cannot compile `sm_90`):

- `python -m pytest -q -m 'not slow'`: `1545 passed, 39 skipped, 117 deselected`;
- `python -m ruff check ...`: passed;
- `git diff --check`: passed.

No package was installed or downloaded. One interrupted validation run had
left the two existing Torch JIT extension caches incomplete; the CUDA LBVH and
direct-light extensions were rebuilt in place with the existing CUDA 12.8
toolchain, then all nine affected tests and the full suite passed.

## Review request

Please review whether:

1. the native-ready evidence is strong enough to prevent cold/allocation or
   converted samples from entering a3 results;
2. submit/resolve is the correct durable split for native CUDA event timing and
   later ordered asynchronous batches;
3. receipt provenance is sufficiently lane-owned and input-bound;
4. the retained/temporary infrastructure boundary still guarantees a clean
   repository after either terminal backend decision.
