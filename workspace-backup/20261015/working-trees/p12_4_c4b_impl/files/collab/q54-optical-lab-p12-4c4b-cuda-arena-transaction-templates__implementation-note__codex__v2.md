# Q54 Optical Lab P12.4 C4b CUDA Arena — Review Corrections

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: corrective implementation note
- Author: Codex
- Version: v2
- Date: 2026-09-09
- Status: ready for re-review
- Corrects: `q54-optical-lab-p12-4c4b-cuda-arena-transaction-templates__implementation-note__codex__v1.md`

## Outcome

The three P1 findings against the initial C4b implementation are closed:

1. an arena transaction cannot activate, enqueue, or finish without an
   arena-issued live lease capability;
2. C3 provenance is derived from and bound to the exact live a2
   `CudaPreparedNativeInput` and `NativeReadinessEvidence`, not caller-supplied
   hashes or separately uploaded tensors;
3. measured `acquire()` performs only fixed-count identity and state checks and
   no longer traverses the sealed slot/frame/resource graph.

The correction does not open a3b, result activation, baseline generation,
host-payload arenas, S, C4d, or B3.

## 1. Arena-only transaction execution

`CudaArenaPlanTarget` no longer exposes a transaction template, executable
stage tuple, or terminal event. The executable target is a separate private
arena-owned object.

When an arena graph is built, each slot receives a private activation and
execution capability. Controlled templates and their stage/event resources are
registered by object identity. Consequently:

- public `activate()` rejects an arena-controlled template;
- directly constructing a transaction from that template is rejected;
- constructing another template around the same controlled stages or event is
  rejected;
- a controlled transaction rejects direct stage enqueue and `finish()`;
- only `CudaActiveArenaSample`, after successful C4 acquire, enters the scoped
  execution authority used by its prebound stages and unique terminal finish.

These checks preserve standalone C2 behavior: non-arena render-only templates
remain publicly activatable.

The negative GPU test exercises public-target inspection, direct template
activation, direct transaction construction, template rebinding, direct stage
enqueue both before and after acquire, and confirms that the C4 slot stays
`FREE` with zero completion records until a legitimate acquire.

## 2. Exact a2 allocation and receipt authority

The canonical a2 preparation functions now issue identity authorities for the
exact prepared object, immutable logical source, and readiness object. CUDA
arena C3 construction requires those live authorities and re-runs untimed
content validation against the source before producing evidence.

The adapter then verifies:

- target packed-scene tensors are the exact a2 tensor objects;
- R0 derived-fixture tensors are the exact a2 tensor objects;
- each dynamic replay input is the approved frame view of the exact a2 replay
  allocation, including device, dtype, shape, stride, base pointer, allocation
  size, and byte offset;
- target manifest and readiness digests match those exact live authorities.

`CudaNativeOpticalArena.bind_c3_provenance()` independently repeats the live
a2 identity/allocation checks. It also requires the exact C3 receipt object
issued by the CUDA arena adapter for that target/prepared/readiness tuple.
Canonical serialization proves receipt contents; adapter issuance proves the
contents were observed from the live allocation graph. Copying or
deserializing an otherwise valid receipt does not inherit that authority.

Negative GPU coverage includes same-content re-upload for dynamic and fixed
inputs, wrong replay storage, copied readiness, correct hashes paired with
mutated device content, mismatched product-side a2 arguments, and a canonical
serialized receipt that lacks adapter issuance.

## 3. Constant-time measured acquire

All graph-wide work remains outside the measured boundary:

- arena sealing validates the complete immutable resource graph;
- `plan_target()` and `prepare_sample()` validate configuration, provenance,
  stages, storage, hashes, and prepare a dormant single-use transaction;
- the arena records an O(1) identity lookup from target to its private
  executable and an O(1) prepared-sample authority record.

Measured `acquire()` now checks only:

- arena seal state;
- one prepared-sample registry lookup;
- fixed object identities for the prepared plan, target, provenance,
  executable, and dormant active sample;
- the current C4 slot/generation/FIFO authority through the C4a control state
  machine.

It does not rebuild signatures, tuples, or resource manifests and does not
iterate slots, replay frames, owners, or resource records. A negative test
replaces all graph-validation helpers with failures immediately before
`acquire()`; valid acquisition and cancellation still succeed.

## Files changed by the correction

- `benchmarks/optical/native.py`
- `benchmarks/optical/receipts.py`
- `benchmarks/optical/cuda_arena.py`
- `optics/cuda_timeline.py`
- `optics/cuda_arena.py`
- `tests/gpu/test_cuda_native_arena.py`
- `PROGRESS.md`
- `REFLECTIONS.md`

## Verification

On `192.168.80.101` in the existing `robot_sim` environment:

- corrected CUDA arena GPU tests: 16 passed;
- C4b/C2/C4a focused unit and GPU set: 73 passed;
- combined C2/C3/C4 unit and GPU set: 122 passed;
- broad `tests/unit/optics` plus `tests/unit/benchmarks`: 446 passed;
- repository-wide pytest: 1816 passed, 115 skipped, with one pre-existing
  pytest fixture deprecation warning;
- Ruff lint and formatting for corrected Python files: passed;
- `git diff --check`: passed.

## Review request

Please review especially:

1. whether controlled template/resource/transaction identities prevent every
   non-arena execution path without restricting standalone C2;
2. whether the two-part proof — canonical receipt contents plus adapter-issued
   object authority — closes both separately uploaded storage and copied
   receipt attacks;
3. whether `acquire()` is now independent of slot count, replay-frame count,
   owner count, and resource-record count;
4. whether the correction remains strictly inside C4b and keeps every result
   and baseline outlet closed.
